#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User-editable configuration
# =============================================================================

OLLAMA_MODELS=(
  "gemma4:12b"
  "gemma4:26b"
)

# Conservative host-memory recommendations for CPU execution of the largest
# configured model. These are warnings, not strict Ollama requirements.
OLLAMA_CPU_MIN_RAM_GIB=32
OLLAMA_CPU_RECOMMENDED_RAM_GIB=48

# Below this amount of GPU memory, Ollama may need to place more model data in
# system RAM. This does not necessarily prevent the experiment from running.
OLLAMA_RECOMMENDED_VRAM_MIB=20000

# Image used for the NVIDIA Container Toolkit smoke test. The official Ollama
# troubleshooting guidance recommends testing `docker run --gpus all ...`.
NVIDIA_TEST_IMAGE="ubuntu:24.04"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

COMPOSE_FILE="${SCRIPT_DIR}/docker/benchmark/docker-compose.yaml"
GPU_COMPOSE_FILE="${SCRIPT_DIR}/docker/benchmark/docker-compose.gpu.yaml"
ENV_FILE="${SCRIPT_DIR}/docker/benchmark/.env"

# =============================================================================
# Runtime state
# =============================================================================

RUN_OPENROUTER=0
RUN_OLLAMA=0
USE_NVIDIA_GPU=0

BENCHMARK_SERVICE=""

BASE_COMPOSE=()
GPU_COMPOSE=()
ACTIVE_COMPOSE=()

# =============================================================================
# Output helpers
# =============================================================================

log() {
  printf '\n==> %s\n' "$*"
}

info() {
  printf 'INFO: %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

confirm() {
  local prompt="$1"
  local answer

  while true; do
    read -r -p "${prompt} [y/N]: " answer

    case "${answer}" in
      y | Y | yes | YES | Yes)
        return 0
        ;;
      "" | n | N | no | NO | No)
        return 1
        ;;
      *)
        printf 'Please answer y or n.\n' >&2
        ;;
    esac
  done
}

# =============================================================================
# Docker checks
# =============================================================================

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: Docker is not installed or the docker command is not in PATH.

Install Docker Engine or Docker Desktop before running this benchmark.

EOF
    exit 1
  fi

  if ! docker version >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: The Docker client is installed, but it cannot communicate with the
Docker daemon.

Possible causes include:

  - Docker is not running.
  - The current user does not have permission to access Docker.
  - DOCKER_HOST points to an unavailable daemon.

Verify the installation with:

  docker info

EOF
    exit 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    cat >&2 <<'EOF'

ERROR: Docker Compose v2 is not available.

This script requires the `docker compose` plugin rather than the legacy
`docker-compose` command.

EOF
    exit 1
  fi

  info "Docker and Docker Compose are available."
}

build_compose_commands() {
  BASE_COMPOSE=(docker compose)

  if [[ -f "${ENV_FILE}" ]]; then
    BASE_COMPOSE+=(--env-file "${ENV_FILE}")
  fi

  BASE_COMPOSE+=(-f "${COMPOSE_FILE}")

  GPU_COMPOSE=(
    "${BASE_COMPOSE[@]}"
    -f "${GPU_COMPOSE_FILE}"
  )

  ACTIVE_COMPOSE=("${BASE_COMPOSE[@]}")
}

validate_compose_files() {
  [[ -f "${COMPOSE_FILE}" ]] ||
    die "Compose file not found: ${COMPOSE_FILE}"

  if ! "${BASE_COMPOSE[@]}" config --quiet; then
    die "The base Docker Compose configuration is invalid."
  fi
}

# =============================================================================
# Experiment selection
# =============================================================================

select_experiments() {
  cat <<'MENU'

Select the experiment to run:

  1) Ollama only

     Starts Neo4j, Ollama, docker-socket-proxy, and the Ollama benchmark
     container.

     Required models are checked and pulled automatically:

       - gemma4:12b
       - gemma4:26b

     When a usable NVIDIA GPU and NVIDIA Container Toolkit are detected, GPU
     acceleration is enabled. Otherwise, the experiment runs on CPU.

  2) OpenRouter only

     Starts Neo4j and the OpenRouter benchmark container.

     Requires OPENROUTER_API_KEY in the current environment or docker/benchmark/.env.

  3) Both Ollama and OpenRouter

  q) Quit

MENU

  while true; do
    read -r -p "Selection [1/2/3/q]: " selection

    case "${selection}" in
      1)
        RUN_OLLAMA=1
        BENCHMARK_SERVICE="benchmark-ollama"
        return
        ;;
      2)
        RUN_OPENROUTER=1
        BENCHMARK_SERVICE="benchmark-openrouter"
        return
        ;;
      3)
        RUN_OPENROUTER=1
        RUN_OLLAMA=1
        BENCHMARK_SERVICE="benchmark-all"
        return
        ;;
      q | Q)
        exit 0
        ;;
      *)
        printf 'Invalid selection. Enter 1, 2, 3, or q.\n' >&2
        ;;
    esac
  done
}

# =============================================================================
# OpenRouter validation
# =============================================================================

env_file_has_nonempty_openrouter_key() {
  local assignment
  local value

  [[ -f "${ENV_FILE}" ]] || return 1

  assignment="$(
    {
      grep -E \
        '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY[[:space:]]*=' \
        "${ENV_FILE}" || true
    } | tail -n 1
  )"

  [[ -n "${assignment}" ]] || return 1

  value="${assignment#*=}"

  value="$(
    printf '%s' "${value}" |
      sed \
        -e 's/^[[:space:]]*//' \
        -e 's/[[:space:]]*$//'
  )"

  # Strip one matching pair of surrounding quotes.
  if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi

  [[ -n "${value}" ]]
}

check_openrouter_key() {
  ((RUN_OPENROUTER == 1)) || return 0

  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    info "OPENROUTER_API_KEY is set in the current environment."
    return 0
  fi

  if env_file_has_nonempty_openrouter_key; then
    info "OPENROUTER_API_KEY was found in ${ENV_FILE}."
    return 0
  fi

  cat >&2 <<EOF

ERROR: The OpenRouter experiment requires OPENROUTER_API_KEY.

Set it in the current shell:

  export OPENROUTER_API_KEY='your-key'

or add it to:

  ${ENV_FILE}

Example:

  OPENROUTER_API_KEY=your-key

EOF

  exit 1
}

# =============================================================================
# Memory detection
# =============================================================================

get_total_ram_kib() {
  if [[ -r /proc/meminfo ]]; then
    awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo
    return
  fi

  if command -v sysctl >/dev/null 2>&1; then
    local bytes

    bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"

    if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$((bytes / 1024))"
      return
    fi
  fi

  return 1
}

get_available_ram_kib() {
  if [[ -r /proc/meminfo ]]; then
    awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo
    return
  fi

  return 1
}

kib_to_gib() {
  awk -v kib="$1" 'BEGIN { printf "%.1f", kib / 1048576 }'
}

report_system_memory() {
  local total_kib
  local available_kib

  total_kib="$(get_total_ram_kib || true)"
  available_kib="$(get_available_ram_kib || true)"

  if [[ "${total_kib}" =~ ^[0-9]+$ ]]; then
    info "Detected system RAM: $(kib_to_gib "${total_kib}") GiB."
  else
    warn "Could not determine total system RAM."
  fi

  if [[ "${available_kib}" =~ ^[0-9]+$ ]]; then
    info "Currently available RAM: $(kib_to_gib "${available_kib}") GiB."
  fi
}

check_cpu_memory() {
  local total_kib
  local available_kib
  local minimum_kib
  local recommended_kib

  total_kib="$(get_total_ram_kib || true)"
  available_kib="$(get_available_ram_kib || true)"

  minimum_kib=$((OLLAMA_CPU_MIN_RAM_GIB * 1024 * 1024))
  recommended_kib=$((OLLAMA_CPU_RECOMMENDED_RAM_GIB * 1024 * 1024))

  if ! [[ "${total_kib}" =~ ^[0-9]+$ ]]; then
    warn \
      "Unable to verify RAM capacity. The ${OLLAMA_MODELS[-1]} model may require substantial system memory in CPU mode."
    return
  fi

  if ((total_kib < minimum_kib)); then
    warn \
      "Only $(kib_to_gib "${total_kib}") GiB of system RAM was detected."

    warn \
      "CPU execution of ${OLLAMA_MODELS[-1]} may fail or cause heavy swapping."

    warn \
      "The configured minimum warning threshold is ${OLLAMA_CPU_MIN_RAM_GIB} GiB."

    if ! confirm "Continue with the CPU-only Ollama experiment?"; then
      exit 1
    fi
  elif ((total_kib < recommended_kib)); then
    warn \
      "$(kib_to_gib "${total_kib}") GiB of RAM was detected. At least ${OLLAMA_CPU_RECOMMENDED_RAM_GIB} GiB is recommended by this script for a more reliable CPU run."
  else
    info "System RAM meets the configured CPU recommendation."
  fi

  if [[ "${available_kib}" =~ ^[0-9]+$ ]] &&
    ((available_kib < minimum_kib)); then

    warn \
      "Only $(kib_to_gib "${available_kib}") GiB is currently available. Close memory-intensive applications before running the 26B model."
  fi
}

# =============================================================================
# NVIDIA GPU and Container Toolkit checks
# =============================================================================

host_has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi -L >/dev/null 2>&1
}

nvidia_toolkit_command_exists() {
  command -v nvidia-ctk >/dev/null 2>&1 ||
    command -v nvidia-container-cli >/dev/null 2>&1
}

docker_reports_nvidia_runtime() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null |
    grep -qi 'nvidia'
}

test_docker_gpu_access() {
  local output

  log "Testing NVIDIA GPU access from Docker."

  if output="$(
    docker run \
      --rm \
      --gpus all \
      "${NVIDIA_TEST_IMAGE}" \
      nvidia-smi 2>&1
  )"; then
    return 0
  fi

  warn "Docker could not create an NVIDIA GPU-enabled test container."

  if [[ -n "${output}" ]]; then
    printf '%s\n' "${output}" |
      tail -n 8 |
      sed 's/^/  /' >&2
  fi

  return 1
}

report_gpu_memory() {
  local max_vram_mib

  max_vram_mib="$(
    nvidia-smi \
      --query-gpu=memory.total \
      --format=csv,noheader,nounits 2>/dev/null |
      tr -d ' ' |
      sort -nr |
      head -n 1
  )"

  if ! [[ "${max_vram_mib}" =~ ^[0-9]+$ ]]; then
    warn "Could not determine NVIDIA GPU memory."
    return
  fi

  info "Largest detected NVIDIA GPU: $((max_vram_mib / 1024)) GiB VRAM."

  if ((max_vram_mib < OLLAMA_RECOMMENDED_VRAM_MIB)); then
    warn \
      "The GPU has less than approximately $((OLLAMA_RECOMMENDED_VRAM_MIB / 1024)) GiB VRAM."

    warn \
      "Ollama may partially offload ${OLLAMA_MODELS[-1]} to system RAM, reducing performance and increasing RAM usage."
  fi
}

assess_ollama_hardware() {
  ((RUN_OLLAMA == 1)) || return 0

  log "Checking hardware for the Ollama experiment."
  report_system_memory

  if ! host_has_nvidia_gpu; then
    warn "No usable NVIDIA GPU was detected."
    info "The Ollama experiment will run in CPU mode."
    check_cpu_memory
    return
  fi

  info "An NVIDIA GPU was detected."

  if nvidia_toolkit_command_exists; then
    info "NVIDIA Container Toolkit commands were found."
  elif docker_reports_nvidia_runtime; then
    info "Docker reports an NVIDIA container runtime."
  else
    warn "The NVIDIA Container Toolkit could not be identified."
  fi

  if test_docker_gpu_access; then
    USE_NVIDIA_GPU=1
    ACTIVE_COMPOSE=("${GPU_COMPOSE[@]}")

    info "NVIDIA GPU acceleration will be enabled."
    report_gpu_memory
    return
  fi

  cat >&2 <<'EOF'

WARNING: An NVIDIA GPU is installed, but Docker cannot expose it to containers.

The NVIDIA Container Toolkit may be missing or may not be configured for
Docker. The benchmark will fall back to CPU execution.

A typical NVIDIA Toolkit configuration uses:

  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

EOF

  check_cpu_memory
}

# =============================================================================
# Service startup
# =============================================================================

validate_active_compose_configuration() {
  if ! "${ACTIVE_COMPOSE[@]}" config --quiet; then
    die "The selected Docker Compose configuration is invalid."
  fi
}

start_services() {
  log "Starting Neo4j."
  "${ACTIVE_COMPOSE[@]}" up -d --wait neo4j

  if ((RUN_OLLAMA == 1)); then
    if ((USE_NVIDIA_GPU == 1)); then
      log "Starting GPU-enabled Ollama and docker-socket-proxy."
    else
      log "Starting CPU-only Ollama and docker-socket-proxy."
    fi

    "${ACTIVE_COMPOSE[@]}" up -d --wait \
      ollama \
      docker-socket-proxy
  fi
}

ensure_ollama_models() {
  ((RUN_OLLAMA == 1)) || return 0

  log "Checking required Ollama models."

  local model

  for model in "${OLLAMA_MODELS[@]}"; do
    if "${ACTIVE_COMPOSE[@]}" exec -T ollama \
      ollama show "${model}" >/dev/null 2>&1; then

      printf 'Ollama model already available: %s\n' "${model}"
    else
      printf 'Pulling missing Ollama model: %s\n' "${model}"

      "${ACTIVE_COMPOSE[@]}" exec -T ollama \
        ollama pull "${model}"
    fi
  done
}

# =============================================================================
# Benchmark execution
# =============================================================================

run_benchmark() {
  log "Starting ${BENCHMARK_SERVICE}."

  # Dependencies are started and checked explicitly above. --no-deps prevents
  # Compose from starting services that are not needed for the selected mode.
  "${ACTIVE_COMPOSE[@]}" run \
    --rm \
    --no-deps \
    -T \
    "${BENCHMARK_SERVICE}" \
    -s -- "${RUN_OPENROUTER}" "${RUN_OLLAMA}" <<'BENCHMARK_SCRIPT'
set -uo pipefail

RUN_OPENROUTER="$1"
RUN_OLLAMA="$2"

OVERALL_STATUS=0

cd /heracles

log() {
  printf '\n==> %s\n' "$*"
}

record_failure() {
  printf '\nERROR: %s\n' "$1" >&2
  OVERALL_STATUS=1
}

# -----------------------------------------------------------------------------
# Load Neo4j
# -----------------------------------------------------------------------------

log "Loading the 3D scene graph into Neo4j."

if ! python /heracles/examples/load_scene_graph.py \
  --scene_graph /heracles/examples/scene_graphs/example_dsg.json; then

  record_failure \
    "Failed to load the Neo4j database. Selected experiments were skipped."
else
  # ---------------------------------------------------------------------------
  # OpenRouter
  # ---------------------------------------------------------------------------

  if [[ "${RUN_OPENROUTER}" == "1" ]]; then
    log "Running the OpenRouter model sweep experiment."

    if ! python examples/experiment_runner.py \
      examples/experiments/openrouter/cypher_model_sweep.yaml \
      examples/experiments/openrouter/pddl_model_sweep.yaml \
      --output-dir output/model_sweep \
      --no-display; then

      record_failure "The OpenRouter experiment failed."
    fi
  fi

  # ---------------------------------------------------------------------------
  # Ollama
  # ---------------------------------------------------------------------------

  if [[ "${RUN_OLLAMA}" == "1" ]]; then
    log "Running the Ollama model sweep experiment."

    if ! python examples/experiment_runner.py \
      examples/experiments/ollama/cypher_model_sweep.yaml \
      examples/experiments/ollama/pddl_model_sweep.yaml \
      --output-dir output/model_sweep \
      --no-display; then

      record_failure "The Ollama experiment failed."
    fi
  fi
fi

# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------
# Always attempt report generation, including after a partial experiment
# failure. Existing results from both providers are included when present.

log "Generating the static HTML results page."

result_directories=(
  output/model_sweep/openrouter/cypher_model_sweep
  output/model_sweep/openrouter/pddl_model_sweep
  output/model_sweep/ollama/cypher_model_sweep
  output/model_sweep/ollama/pddl_model_sweep
)

result_files=()

for directory in "${result_directories[@]}"; do
  [[ -d "${directory}" ]] || continue

  while IFS= read -r -d '' result_file; do
    result_files+=("${result_file}")
  done < <(
    find "${directory}" \
      -maxdepth 1 \
      -type f \
      -name '*_results.yaml' \
      -print0 |
      sort -z
  )
done

if ((${#result_files[@]} == 0)); then
  record_failure \
    "No result YAML files were found; the HTML report could not be generated."
elif ! python examples/display_yaml_results.py \
  "${result_files[@]}" \
  --mode html \
  --output output/model_sweep/report.html; then

  record_failure "Failed to generate the HTML report."
else
  printf '\nReport generated at:\n'
  printf '  /heracles/output/model_sweep/report.html\n'
fi

exit "${OVERALL_STATUS}"
BENCHMARK_SCRIPT
}

# =============================================================================
# Main
# =============================================================================

main() {
  check_docker
  build_compose_commands
  validate_compose_files

  select_experiments
  check_openrouter_key
  assess_ollama_hardware
  validate_active_compose_configuration

  start_services
  ensure_ollama_models
  run_benchmark
}

main "$@"
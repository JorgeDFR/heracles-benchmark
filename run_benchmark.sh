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

OUTPUT_DIR_IN_CONTAINER="output/model_sweep"
BENCHMARK_HOST_OUTPUT_DIR="${BENCHMARK_HOST_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
export BENCHMARK_HOST_OUTPUT_DIR

# =============================================================================
# Runtime state
# =============================================================================

RUN_OPENROUTER=0
RUN_OLLAMA=0
USE_NVIDIA_GPU=0

BENCHMARK_SERVICE=""
SERVICES_STARTED=0
CLEANED_UP=0
SCRIPT_INTERRUPTED=0

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

     Requires OPENROUTER_API_KEY in the current environment or a local
     docker/benchmark/.env file.

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

or add it to this local, ignored file:

  ${ENV_FILE}

Example:

  OPENROUTER_API_KEY=your-key

EOF

  exit 1
}

# =============================================================================
# Cleanup and interruption handling
# =============================================================================

cleanup_services() {
  ((CLEANED_UP == 0)) || return 0
  CLEANED_UP=1

  ((SERVICES_STARTED == 1)) || return 0
  ((${#ACTIVE_COMPOSE[@]} > 0)) || return 0

  if ((SCRIPT_INTERRUPTED == 1)); then
    log "Cleaning up benchmark services after interruption."
  else
    log "Cleaning up benchmark services."
  fi

  "${ACTIVE_COMPOSE[@]}" down --remove-orphans || true
}

on_interrupt() {
  SCRIPT_INTERRUPTED=1
  printf '\nInterrupted, cleaning up...\n' >&2
  cleanup_services
  exit 130
}

install_signal_handlers() {
  trap cleanup_services EXIT
  trap on_interrupt INT TERM
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

prepare_output_directory() {
  mkdir -p "${BENCHMARK_HOST_OUTPUT_DIR}"
  info "Benchmark output will be written under ${BENCHMARK_HOST_OUTPUT_DIR}."
}

check_ollama_container_name_available() {
  ((RUN_OLLAMA == 1)) || return 0

  local owner_project

  if ! docker container inspect ollama >/dev/null 2>&1; then
    return 0
  fi

  owner_project="$(
    docker inspect \
      -f '{{ index .Config.Labels "com.docker.compose.project" }}' \
      ollama 2>/dev/null || true
  )"

  if [[ "${owner_project}" == "heracles-benchmark" ]]; then
    return 0
  fi

  cat >&2 <<EOF

ERROR: A Docker container named "ollama" already exists and does not appear to
belong to the heracles-benchmark Compose project.

The Ollama local metrics configuration expects the benchmark Ollama container
to use this name. Stop or rename the existing container before running the
Ollama benchmark.

EOF

  exit 1
}

pre_start_cleanup() {
  log "Removing stale benchmark containers, if any."
  "${ACTIVE_COMPOSE[@]}" down --remove-orphans || true
}

start_services() {
  SERVICES_STARTED=1

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
  log "Running ${BENCHMARK_SERVICE}."

  "${ACTIVE_COMPOSE[@]}" run \
    --rm \
    --build \
    --no-deps \
    "${BENCHMARK_SERVICE}" \
    /home/benchmark/workspace/benchmark_runner.sh \
    "${RUN_OPENROUTER}" \
    "${RUN_OLLAMA}" \
    "${OUTPUT_DIR_IN_CONTAINER}"

  printf '\nBenchmark report path on the host:\n'
  printf '  %s/model_sweep/report.html\n' "${BENCHMARK_HOST_OUTPUT_DIR}"
}

set_host_user_ids() {
  if ! command -v id >/dev/null 2>&1; then
    die "The id command is required to determine the host UID and GID."
  fi

  export HOST_UID
  export HOST_GID

  HOST_UID="$(id -u)"
  HOST_GID="$(id -g)"

  if [[ "${HOST_UID}" == "0" || "${HOST_GID}" == "0" ]]; then
    warn "The benchmark image is being built using root UID or GID."
  fi

  info "Benchmark container UID:GID will be ${HOST_UID}:${HOST_GID}."
}

# =============================================================================
# Main
# =============================================================================

main() {
  check_docker
  set_host_user_ids
  build_compose_commands
  validate_compose_files

  select_experiments
  check_openrouter_key
  assess_ollama_hardware
  validate_active_compose_configuration
  install_signal_handlers
  prepare_output_directory
  check_ollama_container_name_available
  pre_start_cleanup

  start_services
  ensure_ollama_models
  run_benchmark
}

main "$@"

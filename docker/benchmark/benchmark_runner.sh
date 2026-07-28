#!/usr/bin/env bash
set -Eeuo pipefail

RUN_OPENROUTER="${1:-0}"
RUN_OLLAMA="${2:-0}"
OUTPUT_DIR="${3:-output/model_sweep}"

OVERALL_STATUS=0
WORKSPACE_DIR="/home/benchmark/workspace"

cd "${WORKSPACE_DIR}/heracles_agents"

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 2
}

record_failure() {
  printf '\nERROR: %s\n' "$1" >&2
  OVERALL_STATUS=1
}

validate_flag() {
  local name="$1"
  local value="$2"

  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    die "${name} must be 0 or 1; got '${value}'."
  fi
}

validate_flag RUN_OPENROUTER "${RUN_OPENROUTER}"
validate_flag RUN_OLLAMA "${RUN_OLLAMA}"

if [[ "${RUN_OPENROUTER}" == "0" && "${RUN_OLLAMA}" == "0" ]]; then
  die "At least one provider must be selected."
fi

log "Loading the 3D scene graph into Neo4j."

if ! python "${WORKSPACE_DIR}/heracles/examples/load_scene_graph.py" \
  --scene_graph "${WORKSPACE_DIR}/heracles/examples/scene_graphs/example_dsg.json"; then

  record_failure \
    "Failed to load the Neo4j database. Selected experiments were skipped."
else
  if [[ "${RUN_OPENROUTER}" == "1" ]]; then
    log "Running the OpenRouter model sweep experiment."

    if ! python examples/experiment_runner.py \
      examples/experiments/openrouter/cypher_model_sweep.yaml \
      examples/experiments/openrouter/pddl_model_sweep.yaml \
      --output-dir "${OUTPUT_DIR}" \
      --no-display; then

      record_failure "The OpenRouter experiment failed."
    fi
  fi

  if [[ "${RUN_OLLAMA}" == "1" ]]; then
    log "Running the Ollama model sweep experiment."

    if ! python examples/experiment_runner.py \
      examples/experiments/ollama/cypher_model_sweep.yaml \
      examples/experiments/ollama/pddl_model_sweep.yaml \
      --output-dir "${OUTPUT_DIR}" \
      --no-display; then

      record_failure "The Ollama experiment failed."
    fi
  fi
fi

log "Generating the static HTML results page."

result_directories=()

if [[ "${RUN_OPENROUTER}" == "1" ]]; then
  result_directories+=(
    "${OUTPUT_DIR}/openrouter/cypher_model_sweep"
    "${OUTPUT_DIR}/openrouter/pddl_model_sweep"
  )
fi

if [[ "${RUN_OLLAMA}" == "1" ]]; then
  result_directories+=(
    "${OUTPUT_DIR}/ollama/cypher_model_sweep"
    "${OUTPUT_DIR}/ollama/pddl_model_sweep"
  )
fi

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
  --output "${OUTPUT_DIR}/report.html"; then

  record_failure "Failed to generate the HTML report."
else
  printf '\nReport generated at:\n'
  printf '  %s/heracles_agents/%s/report.html\n' \
    "${WORKSPACE_DIR}" \
    "${OUTPUT_DIR}"
fi

exit "${OVERALL_STATUS}"

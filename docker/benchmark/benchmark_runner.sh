#!/usr/bin/env bash
set -Eeuo pipefail

RUN_OPENROUTER="${1:-0}"
RUN_OLLAMA="${2:-0}"
BENCHMARK_CONFIG="${3:-/home/benchmark/workspace/benchmark/configs/benchmark.yaml}"

OVERALL_STATUS=0
WORKSPACE_DIR="/home/benchmark/workspace"
BENCHMARK_ROOT="${WORKSPACE_DIR}/benchmark"
MANIFEST_TOOL="${BENCHMARK_ROOT}/scripts/benchmark_manifest.py"
CONSOLE_TOOL="${BENCHMARK_ROOT}/scripts/benchmark_console.py"
RUNTIME_DIR="$(mktemp -d /tmp/heracles-benchmark-runtime.XXXXXX)"

cd "${WORKSPACE_DIR}/heracles_agents"

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

providers=()

if [[ "${RUN_OPENROUTER}" == "1" ]]; then
  providers+=(openrouter)
fi

if [[ "${RUN_OLLAMA}" == "1" ]]; then
  providers+=(ollama)
fi

prepare_args=(
  --config "${BENCHMARK_CONFIG}"
  prepare
  --runtime-dir "${RUNTIME_DIR}"
)

for provider in "${providers[@]}"; do
  prepare_args+=(--provider "${provider}")
done

SCENE_GRAPH="$(
  python "${MANIFEST_TOOL}" \
    --config "${BENCHMARK_CONFIG}" \
    value scene_graph
)"
OUTPUT_DIR="$(
  python "${MANIFEST_TOOL}" \
    --config "${BENCHMARK_CONFIG}" \
    value output_dir
)"

mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/benchmark.log"

if ! python "${CONSOLE_TOOL}" run \
  --label "Preparing benchmark configuration" \
  --log-file "${RUN_LOG}" \
  --reset-log \
  -- \
  python "${MANIFEST_TOOL}" "${prepare_args[@]}"; then

  die "The benchmark manifest is invalid."
fi

cp \
  "${RUNTIME_DIR}/benchmark_manifest.resolved.yaml" \
  "${OUTPUT_DIR}/benchmark_manifest.resolved.yaml"
RUN_MARKER="${RUNTIME_DIR}/run-started"
touch "${RUN_MARKER}"

if [[ "${RUN_OLLAMA}" == "1" ]]; then
  if ! python "${CONSOLE_TOOL}" run \
    --label "Preparing configured Ollama models" \
    --log-file "${RUN_LOG}" \
    -- \
    python "${MANIFEST_TOOL}" \
      --config "${BENCHMARK_CONFIG}" \
      pull-ollama \
      --host "${OLLAMA_HOST:-http://ollama:11434}"; then

    die "The configured Ollama models could not be prepared."
  fi
fi

if ! python "${CONSOLE_TOOL}" run \
  --label "Loading scene graph into Neo4j" \
  --kind scene \
  --log-file "${RUN_LOG}" \
  -- \
  python "${WORKSPACE_DIR}/heracles/examples/load_scene_graph.py" \
    --scene_graph "${SCENE_GRAPH}"; then

  record_failure \
    "Failed to load the Neo4j database. Selected experiments were skipped."
else
  for provider in "${providers[@]}"; do
    experiment_paths=(
      "${RUNTIME_DIR}/${provider}/cypher_model_sweep.yaml"
      "${RUNTIME_DIR}/${provider}/pddl_model_sweep.yaml"
    )

    if ! python "${CONSOLE_TOOL}" experiments \
      --provider "${provider}" \
      --runner "${WORKSPACE_DIR}/heracles_agents/examples/experiment_runner.py" \
      --output-dir "${OUTPUT_DIR}" \
      --log-file "${RUN_LOG}" \
      --experiment "${experiment_paths[0]}" \
      --experiment "${experiment_paths[1]}"; then

      record_failure "The ${provider} experiment failed."
    fi
  done
fi

result_files=()

while IFS= read -r -d '' result_file; do
  result_files+=("${result_file}")
done < <(
  find "${OUTPUT_DIR}" \
    -type f \
    -name '*_results.yaml' \
    -newer "${RUN_MARKER}" \
    -print0 |
    sort -z
)

REPORT_PATH="${OUTPUT_DIR}/report.html"

if ((${#result_files[@]} == 0)); then
  record_failure \
    "No result YAML files were found; the HTML report could not be generated."
elif ! python "${CONSOLE_TOOL}" run \
  --label "Generating HTML report" \
  --success-detail "${REPORT_PATH}" \
  --log-file "${RUN_LOG}" \
  -- \
  python examples/display_yaml_results.py \
    "${result_files[@]}" \
    --mode html \
    --output "${REPORT_PATH}"; then

  record_failure "Failed to generate the HTML report."
fi

SUMMARY_TITLE="Benchmark complete"
if ((OVERALL_STATUS != 0)); then
  SUMMARY_TITLE="Benchmark finished with errors"
fi

python "${CONSOLE_TOOL}" summary \
  --title "${SUMMARY_TITLE}" \
  --field "Manifest=${BENCHMARK_CONFIG}" \
  --field "Output=${OUTPUT_DIR}" \
  --field "Report=${REPORT_PATH}" \
  --field "Detailed log=${RUN_LOG}"

exit "${OVERALL_STATUS}"

# Heracles Benchmark Docker Workflow

The Docker benchmark reads all run-specific inputs from a repository-owned YAML
manifest. The default is [`configs/benchmark.yaml`](../../configs/benchmark.yaml).
The shell launcher only handles provider selection, host checks, services, and
cleanup.

## Prerequisites

- Docker Engine or Docker Desktop
- Docker Compose v2, available as `docker compose`
- Bash
- initialized `external/heracles` and `external/heracles_agents` submodules
- for Ollama GPU runs, NVIDIA drivers and NVIDIA Container Toolkit
- for OpenRouter runs, an OpenRouter API key

When an NVIDIA GPU is detected, the launcher tests Docker GPU access. A
successful check enables `docker-compose.gpu.yaml`; otherwise Ollama falls back
to CPU mode after a host-memory warning.

## Configure a run

Edit or copy `configs/benchmark.yaml`. It contains the scene graph, QA and PDDL
files, question metadata, output directory, agent settings, provider switches,
model lists, and metric settings. Set `enabled: true` only for the models that
should run.

All manifest paths are relative to the repository root. Keep custom manifests
below `configs/`; they are copied into the benchmark image together with
`data/`. The manifest validator rejects missing files, paths outside the
repository, output paths outside `output/`, disabled selected providers, empty
model sweeps, and question metadata whose scene or checksums do not match.

Validate before starting Docker from a local virtual environment:

```bash
python scripts/benchmark_manifest.py \
  --config configs/benchmark.yaml \
  validate
```

The question generator convention is:

```text
data/questions/<scene-id>/
├── qa_questions.yaml
├── pddl_questions.yaml
└── metadata.yaml
```

`metadata.yaml` records the scene and question checksums, seeds, dependency
versions, generator version, and question counts. The Docker workflow verifies
these values before loading the graph or running a model. It is the sole
metadata file; `qa_questions.yaml` and `pddl_questions.yaml` contain only their
question lists.

## Environment setup

OpenRouter benchmarks require `OPENROUTER_API_KEY`. Export it:

```bash
export OPENROUTER_API_KEY='your-key'
```

Alternatively, create the ignored environment file:

```bash
cp docker/benchmark/.env.example docker/benchmark/.env
```

Then add the key to `docker/benchmark/.env`. Do not commit this file.

The launcher exports:

- `HOST_UID` and `HOST_GID`, so generated output belongs to the host user;
- `BENCHMARK_HOST_OUTPUT_DIR`, which defaults to `<repo>/output` and is mounted
  at `/home/benchmark/workspace/benchmark/output` in the container.

The manifest's `benchmark.output_dir` must be within that mounted `output/`
tree. To use a different host storage location:

```bash
export BENCHMARK_HOST_OUTPUT_DIR="$PWD/output"
./scripts/run_benchmark.sh
```

## Run the benchmark

Use the default manifest:

```bash
./scripts/run_benchmark.sh
```

Or select another repository manifest:

```bash
./scripts/run_benchmark.sh --config configs/my-benchmark.yaml
```

The `BENCHMARK_CONFIG` environment variable provides the same override. The
launcher presents an interactive choice of Ollama, OpenRouter, both, or quit.
The selected provider must also be enabled in the manifest.

The benchmark container displays Rich progress bars for long-running model
sweeps, followed by a normalized table containing the model, task, question
count, valid answers, correct answers, and accuracy. Request-level HTTP logs,
Neo4j notifications, answer-parser warnings, and raw tool errors are suppressed
from the live console and written to `benchmark.log`. A diagnostics table shows
how many messages of each category were captured, so the cleaner output does not
hide their existence.

Docker Compose build and service progress defaults to quiet mode. Set
`COMPOSE_PROGRESS=auto` before launching if the full Docker progress display is
useful for troubleshooting.

For Ollama runs, enabled model names are read from the manifest and missing
models are pulled into the persistent `ollama-cache` volume. If an unrelated
container named `ollama` already exists, stop or rename it because local metric
collection expects the benchmark container to use that name.

For the default manifest, outputs are stored under:

```text
<repo>/output/example_dsg/model_sweep/
```

That directory contains provider results, `report.html`,
`benchmark_manifest.resolved.yaml`, and a private-permission `benchmark.log`
with complete subprocess diagnostics. The resolved manifest captures the
source manifest and its checksum, selected providers, all input checksums,
question generation metadata (including seeds and dependency versions), and
the full configuration used for the run.

## Stop and clean up

Press `Ctrl+C` to stop. The launcher runs:

```bash
docker compose ... down --remove-orphans
```

It exits with status 130 after an interrupt. Cleanup does not pass `-v`, so the
Ollama model cache is preserved.

## Manual validation

If `docker/benchmark/.env` does not exist, substitute `.env.example` below:

```bash
export BENCHMARK_HOST_OUTPUT_DIR="$PWD/output"

docker compose \
  --env-file docker/benchmark/.env \
  -f docker/benchmark/docker-compose.yaml \
  config --quiet

docker compose \
  --env-file docker/benchmark/.env \
  -f docker/benchmark/docker-compose.yaml \
  -f docker/benchmark/docker-compose.gpu.yaml \
  config --quiet

bash -n scripts/run_benchmark.sh docker/benchmark/benchmark_runner.sh
python -m unittest discover -s tests -v
```

## Troubleshooting

- Missing OpenRouter key: export it or add it to the ignored benchmark `.env`.
- Invalid manifest: run `scripts/benchmark_manifest.py ... validate` for the
  exact path, metadata, provider, or model error.
- Docker cannot expose the GPU: configure NVIDIA Container Toolkit and restart
  Docker; the run can fall back to CPU.
- CPU run warns about RAM: adjust the launcher thresholds for the selected
  models or use a host with more memory.
- Existing `ollama` conflict: stop or rename the non-benchmark container.
- Stale containers: rerun the launcher; it performs a pre-start
  `down --remove-orphans`.

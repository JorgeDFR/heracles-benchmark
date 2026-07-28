# Heracles Benchmark Docker Workflow

This folder contains the Docker image and Compose services used by the root
benchmark launcher:

```bash
./run_benchmark.sh
```

Run the script from the repository root. The script presents an interactive
menu for:

- Ollama only
- OpenRouter only
- both Ollama and OpenRouter

The script starts the required Docker services, runs the selected benchmark
container, writes results under the configured host output directory, and cleans
up transient containers and networks when it exits. The `ollama-cache` Docker
volume is intentionally preserved so pulled models are reused across runs.

## Prerequisites

- Docker Engine or Docker Desktop
- Docker Compose v2, available as `docker compose`
- Bash
- For Ollama GPU runs: NVIDIA drivers, `nvidia-smi`, and NVIDIA Container
  Toolkit configured for Docker
- For OpenRouter runs: an OpenRouter API key

When an NVIDIA GPU is detected, the launcher tests Docker GPU access. If the
test succeeds, it adds `docker-compose.gpu.yaml`; otherwise the Ollama benchmark
falls back to CPU mode after warning about host memory.

## Environment Setup

OpenRouter benchmarks require `OPENROUTER_API_KEY`. Prefer exporting it in the
current shell:

```bash
export OPENROUTER_API_KEY='your-key'
```

Alternatively, create a local ignored env file from the template:

```bash
cp docker/benchmark/.env.example docker/benchmark/.env
```

Then edit `docker/benchmark/.env`:

```env
COMPOSE_PROJECT_NAME=heracles-benchmark
OPENROUTER_API_KEY=your-key
```

Do not commit `docker/benchmark/.env`; it is for local secrets and machine-local
settings only.

The launcher also exports these values automatically:

- `HOST_UID` and `HOST_GID`: used so benchmark output files are owned by your
  host user.
- `BENCHMARK_HOST_OUTPUT_DIR`: defaults to `<repo>/output` and is bind-mounted
  to `/home/benchmark/workspace/heracles_agents/output` in the benchmark
  container.

To write results somewhere else, export `BENCHMARK_HOST_OUTPUT_DIR` before
running the launcher:

```bash
export BENCHMARK_HOST_OUTPUT_DIR="$PWD/heracles_agents/output"
./run_benchmark.sh
```

## Running The Benchmark

From the repository root:

```bash
./run_benchmark.sh
```

Choose one menu option:

```text
1) Ollama only
2) OpenRouter only
3) Both Ollama and OpenRouter
q) Quit
```

For Ollama runs, the script checks and pulls these models if missing:

- `gemma4:12b`
- `gemma4:26b`

If a non-benchmark Docker container named `ollama` already exists, stop or rename
it before running the Ollama benchmark. Local metrics expect the benchmark
Ollama container to use that name.

## Stopping A Run

Press `Ctrl+C` to stop the script. The launcher handles the interrupt, runs
Compose cleanup once, and exits with status `130`.

Cleanup uses:

```bash
docker compose ... down --remove-orphans
```

It does not pass `-v`, so the `ollama-cache` volume is not deleted.

## Outputs

Benchmark YAML files and the HTML report are written under:

```text
${BENCHMARK_HOST_OUTPUT_DIR}/model_sweep
```

By default, that resolves to:

```text
<repo>/output/model_sweep
```

The combined HTML report is:

```text
${BENCHMARK_HOST_OUTPUT_DIR}/model_sweep/report.html
```

Inside the container, the same output directory is:

```text
/home/benchmark/workspace/heracles_agents/output/model_sweep
```

The repository keeps the top-level `output` directory via `output/.gitkeep`, but
generated files inside `output/` are ignored by Git.

## Manual Validation

The launcher validates Docker and Compose before running. If you want to inspect
the Compose configuration manually, run these commands from the repository root:

```bash
docker compose \
  --env-file docker/benchmark/.env \
  -f docker/benchmark/docker-compose.yaml \
  config --quiet

docker compose \
  --env-file docker/benchmark/.env \
  -f docker/benchmark/docker-compose.yaml \
  -f docker/benchmark/docker-compose.gpu.yaml \
  config --quiet
```

If you do not have `docker/benchmark/.env`, omit the `--env-file` arguments and
export any required variables in your shell instead.

Non-Docker checks:

```bash
bash -n run_benchmark.sh docker/benchmark/benchmark_runner.sh
shellcheck run_benchmark.sh docker/benchmark/benchmark_runner.sh
```

## Troubleshooting

- `OPENROUTER_API_KEY` missing: export it in the shell or add it to the local
  ignored `docker/benchmark/.env` file.
- Docker cannot expose the GPU: configure NVIDIA Container Toolkit for Docker,
  then restart Docker. The script prints the typical `nvidia-ctk` command when
  this check fails.
- CPU-only Ollama run warns about RAM: close memory-heavy applications or use a
  host with more RAM before running the 26B model.
- Existing `ollama` container conflict: stop or rename the existing container so
  the benchmark Compose project can create its own `ollama` container.
- Stale benchmark containers: rerun `./run_benchmark.sh`; it performs a
  pre-start `down --remove-orphans` for the benchmark Compose project.

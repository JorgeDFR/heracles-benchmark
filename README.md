# Heracles Benchmark

This repository contains the reproducible question-generation and Docker
benchmark workflows for Heracles. It owns the benchmark inputs and settings;
the Heracles implementations are kept as Git submodules under `external/`.

## Repository layout

```text
.
├── configs/
│   └── benchmark.yaml        # Scene, questions, agent settings, and model sweeps
├── data/
│   ├── questions/
│   │   ├── example_dsg/      # Questions and metadata grouped by source scene
│   │   └── question_types.yaml
│   └── scene_graphs/         # Input Spark DSG JSON files
├── docker/
│   ├── benchmark/            # Reproducible model-sweep workflow
│   ├── devel/                # Interactive development services
│   └── main/                 # Standalone application images
├── external/
│   ├── heracles/             # Git submodule
│   └── heracles_agents/      # Git submodule
├── output/                   # Generated benchmark reports; ignored by Git
├── scripts/                  # Repository-level Python and shell entry points
└── requirements.txt          # Dependencies for local Python utilities
```

## Clone and initialize

```bash
git clone --recurse-submodules \
  https://github.com/JorgeDFR/heracles-benchmark.git
cd heracles-benchmark
```

For an existing clone without initialized submodules, or after pulling a
commit that changes their revisions, run:

```bash
git submodule update --init --recursive
```

## Local Python setup

Use a repository-local virtual environment for scripts under `scripts/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate questions from the bundled graph:

```bash
python scripts/generate_questions.py \
  data/scene_graphs/example_dsg.json \
  --seed 7
```

The default output directory is derived from the scene filename, so this writes
the following files under `data/questions/example_dsg/`:

- `qa_questions.yaml`
- `pddl_questions.yaml`
- `metadata.yaml`

The metadata records scene, catalog, generator, avoidance-input, and output
checksums; generator and dependency versions; random seeds; and question
counts. The QA and PDDL files contain only their `questions` lists; all metadata
is centralized in `metadata.yaml`. If two graphs have the same filename stem,
assign a stable identifier with `--scene-id`; the default output folder will use
that identifier. An explicit `--output-dir` must have the same final directory
name as the scene ID.

The generator produces one grounded item per entry in
`data/questions/question_types.yaml`: 50 QA types and 50 PDDL types. Generation
is deterministic for a given graph, catalog, and seed. Run
`python scripts/generate_questions.py --list-types` to inspect the taxonomy.

Use `--avoid-questions path/to/questions.yaml` one or more times to reject exact
wording collisions with additional benchmark, evaluation, or training files.
The example questions under `external/heracles_agents` are checked by default.

The full suite expects semantic object and room labels, at least two objects and
rooms, 2D or 3D navigable places, containment edges, and connected room and
place pairs. Unsupported scene graphs fail with an explanation instead of
producing ungrounded answers.

## Benchmark configuration

[`configs/benchmark.yaml`](configs/benchmark.yaml) is the single source of
truth for a Docker benchmark run. It declares:

- the scene ID and repository-owned scene graph;
- the QA, PDDL, and question-metadata files;
- the output directory;
- agent temperature, seed, and iteration limit;
- enabled providers and each provider's model sweep;
- Ollama local-metric settings.

Paths are relative to this repository. Input paths must exist, the output must
be below `output/`, and question metadata must match the scene and question-file
checksums. Validate a manifest locally with:

```bash
python scripts/benchmark_manifest.py \
  --config configs/benchmark.yaml \
  validate
```

Copy the manifest to create another benchmark configuration. Keep manifests
under `configs/`, scene graphs under `data/scene_graphs/`, and each generated
question set under `data/questions/<scene-id>/`.

## Run the Docker benchmark

Install Docker Engine or Docker Desktop with Docker Compose v2, and initialize
the Git submodules before building. Then run:

```bash
cp docker/benchmark/.env.example docker/benchmark/.env
./scripts/run_benchmark.sh
```

Choose Ollama, OpenRouter, or both from the menu. For OpenRouter, set
`OPENROUTER_API_KEY` in the shell or in the ignored
`docker/benchmark/.env` file. To use a different manifest:

```bash
./scripts/run_benchmark.sh --config configs/my-benchmark.yaml
```

The launcher builds the selected repository inputs into the image, resolves
provider experiment files from the manifest, pulls enabled Ollama models when
needed, loads the configured graph, and writes results to the manifest's
`benchmark.output_dir`. A resolved manifest is saved beside the results.
The console uses Rich progress bars and compact per-model result tables. Noisy
HTTP, database-notification, validation, and tool diagnostics are retained in
`benchmark.log` beside the report instead of being streamed to the terminal.

See [docker/benchmark/README.md](docker/benchmark/README.md) for GPU detection,
memory guidance, output mounts, cleanup behavior, and troubleshooting.

## Other Docker workflows

Launch Neo4j, Ollama, and the interactive Heracles CLI for development with:

```bash
cp docker/devel/.env.example docker/devel/.env
./scripts/run_heracles.sh
```

The development configuration enables NVIDIA GPU access for Ollama. The
standalone services under `docker/main/` use the same repository layout:

```bash
cp docker/main/.env.example docker/main/.env
docker compose \
  --env-file docker/main/.env \
  -f docker/main/docker-compose.yaml \
  up -d neo4j
docker compose \
  --env-file docker/main/.env \
  -f docker/main/docker-compose.yaml \
  run --rm cli
```

Set `OPENROUTER_API_KEY` in `docker/main/.env` before running `chatdsg`.

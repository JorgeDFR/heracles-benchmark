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

<details>
<summary>Question-generation details and scene requirements</summary>

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

Each catalog entry owns its `question_template`, with named placeholders such
as `{center_obj}` filled by the generator. Edit wording in the catalog; graph
sampling and solution computation remain in Python. Display names are derived
from the semantic IDs rather than maintained separately in the catalog.

The `overlap_class` field classifies each type against the 3dsg-cypher-agent
QA/PDDL template catalogs: `direct` for the same central task, `related` for a
variant or composition, and `absent` for an operation not found there. These
labels describe task-family overlap, not confirmed model training exposure.
Generated questions retain `question_type`, `overlap_class`, and operation
tags. The HTML report's **Task overlap** selector filters Questions and
recomputes Overview metrics for the selected class. Older results without
classification appear as **Unclassified**.

Use `--avoid-questions path/to/questions.yaml` one or more times to reject
wording collisions with additional benchmark, evaluation, or training files.
The check includes parameterized wording variants such as `{case.object}` and
checks both catalog templates and rendered questions. The example questions
under `external/heracles_agents` are checked by default. Both training catalogs
are also checked automatically when the sibling `3dsg-cypher-agent` checkout
is available; otherwise pass their paths explicitly. Avoidance inputs and their
checksums are recorded in the generated metadata.

The full suite expects semantic object and room labels, at least two objects and
rooms, 2D or 3D navigable places, containment edges, and connected room and
place pairs. Unsupported scene graphs fail with an explanation instead of
producing ungrounded answers.

</details>

## Benchmark configuration

[`configs/benchmark.yaml`](configs/benchmark.yaml) is the single source of
truth for a Docker benchmark run. It declares:

- the scene ID and repository-owned scene graph;
- the QA, PDDL, and question-metadata files;
- the output directory;
- normalized reasoning, temperature, seed, and iteration settings;
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

<details>
<summary>Inference controls and model-capability inspection</summary>

Inference controls are defined once under `agent` and may be replaced for an
individual model with a `parameters` mapping:

```yaml
agent:
  temperature: 0.2
  seed: 123
  reasoning:
    mode: enabled
    effort: model_default         # `null` and an omitted effort are equivalent.

providers:
  openrouter:
    require_parameters: true
    models:
      - alias: example
        model: provider/model
        capabilities:               # Declared and validated for every enabled model.
          reasoning: unsupported    # required, optional, or unsupported
          reasoning_efforts: null   # Provider effort names, or null if unavailable.
          temperature: false
          seed: false
        parameters:                 # Optional model-specific replacement.
          temperature: null         # Omit when unsupported.
          seed: null                # Omit when unsupported.
          reasoning:
            mode: unsupported
            effort: null
```

For OpenRouter, optionally pin each model to a serving provider or endpoint:

```yaml
parameters:
  openrouter_provider: deepinfra  # Or an exact endpoint slug, e.g. deepinfra/turbo.
```

This restricts requests to that provider and disables provider fallbacks, so an
unavailable provider fails the request instead of switching to a backend with
different cost or throughput. Use the provider/endpoint slug from the model's
OpenRouter page; a base slug can match multiple endpoints. See
[OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).
Omit the setting or use `null` to retain automatic routing. The setting is only
supported for OpenRouter models and is saved in result metadata and shown in the
report's Provider Models tab. The global `require_parameters` policy still applies.

Reasoning mode is `enabled`, `disabled`, or `unsupported`. Use `unsupported`
for a model with no reasoning control. An enabled model may use any non-empty
provider effort name. Set `effort: model_default`, `effort: null`, or omit the
field to enable reasoning while using the model/provider default, whether or
not explicit effort levels are available. `model_default` is canonicalized to
null before execution. A per-model override lets other models retain a global
effort:

```yaml
capabilities:
  reasoning: optional
  reasoning_efforts: null
  temperature: true
  seed: true
parameters:
  reasoning:
    mode: enabled
    effort: model_default
```

Models with mandatory reasoning must use `enabled`. When an explicit effort is
selected, it must be supported by that model. For example, Ollama GPT-OSS
cannot disable thinking and accepts only `low`,
`medium`, or `high`. OpenRouter's `require_parameters: true` prevents routing
to an endpoint that would silently ignore supplied controls. The requested
effective settings and declared capabilities are saved in result metadata and
shown in the report's Provider Models tab. Capabilities are kept explicitly in
the manifest rather than fetched during a run, so validation does not add a
network request or change when a provider updates its model catalog.
When an effort is provided, validation checks it against the model capability
declaration. Supported temperature and seed controls still require explicit
values for reproducibility.

Inspect an exact provider model and generate a suggested capability block with:

```bash
python scripts/inspect_model_capabilities.py openrouter openai/gpt-oss-120b
python scripts/inspect_model_capabilities.py ollama gemma4:26b
```

The command retrieves OpenRouter catalog metadata or the local Ollama
`/api/show` response. It also lists fields that the API cannot establish and
links to the model-specific documentation where they should be confirmed. Use
`--format yaml` or `--format json` for machine-readable output, and
`--ollama-host` when Ollama is not available at `http://localhost:11434`.
When the default local Ollama address is offline, the command automatically
starts the repository's `ollama` Compose service and waits for its API. Pass
`--no-start-ollama` to disable this behavior. The requested model must already
exist in the persistent `ollama-cache` volume; otherwise install it with
`docker exec ollama ollama pull <model>` and run the inspection again.

</details>

## Run the Docker benchmark

Install Docker Engine or Docker Desktop with Docker Compose v2, and initialize
the Git submodules before building. Then run:

```bash
cp docker/benchmark/.env.example docker/benchmark/.env
```

```bash
./scripts/run_benchmark.sh
```

Choose Ollama, OpenRouter, or both from the menu. For OpenRouter, set
`OPENROUTER_API_KEY` in the shell or in the ignored
`docker/benchmark/.env` file. To use a different manifest:

```bash
./scripts/run_benchmark.sh --config configs/my-benchmark.yaml
```

<details>
<summary>Benchmark execution, logging, and Ollama warmup details</summary>

The launcher builds the selected repository inputs into the image, resolves
provider experiment files from the manifest, pulls enabled Ollama models when
needed, loads the configured graph, and writes results to the manifest's
`benchmark.output_dir`. A resolved manifest is saved beside the results.
The console uses Rich progress bars and compact per-model result tables. Noisy
HTTP, database-notification, validation, and tool diagnostics are retained in
`benchmark.log` beside the report instead of being streamed to the terminal.

For Ollama, the launcher-generated configuration collects the unloaded GPU
baseline, performs one separately recorded preload/cold-start request followed
by the configured number of post-load warmup requests, and only then starts
measured questions. `keep_alive` keeps the model resident, and `/api/ps`
verifies residency before measurement. Set
`warmup_enabled`, `warmup_requests`, `warmup_prompt`, `warmup_keep_alive`, and
`warmup_verify_resident` under `providers.ollama.local_metrics`.

</details>

<p></p>

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

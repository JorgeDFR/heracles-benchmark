#!/usr/bin/env python3
"""Validate and materialize repository-owned Heracles benchmark manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PROVIDERS = ("ollama", "openrouter")
TASK_SPECS = {
    "qa": {
        "filename": "cypher_model_sweep.yaml",
        "task": "cypher",
        "name": "cypher",
        "configuration": "agentic-cypher-qa-{alias}",
        "result_configuration": "agentic-cypher-qa",
        "prompt": "$HERACLES_AGENTS_PATH/examples/prompts/cypher/qa_agentic_cypher_prompt.yaml",
        "output_type": "SLDP",
    },
    "pddl": {
        "filename": "pddl_model_sweep.yaml",
        "task": "pddl",
        "name": "pddl",
        "configuration": "agentic-cypher-pddl-{alias}",
        "result_configuration": "agentic-cypher-pddl",
        "prompt": "$HERACLES_AGENTS_PATH/examples/prompts/cypher/pddl_agentic_cypher_prompt.yaml",
        "output_type": "PDDL",
    },
}


class ManifestError(ValueError):
    """A benchmark manifest is incomplete, unsafe, or internally inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"`{field}` must be a mapping")
    return value


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"`{field}` must be a non-empty string")
    return value.strip()


@dataclass
class BenchmarkManifest:
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> BenchmarkManifest:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ManifestError(f"Benchmark manifest does not exist: {resolved}")
        try:
            resolved.relative_to(ROOT)
        except ValueError as error:
            raise ManifestError(
                f"Benchmark manifest must be inside the repository: {resolved}"
            ) from error
        with resolved.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        return cls(resolved, required_mapping(raw, "manifest"))

    @property
    def benchmark(self) -> dict[str, Any]:
        return required_mapping(self.raw.get("benchmark"), "benchmark")

    @property
    def agent(self) -> dict[str, Any]:
        return required_mapping(self.raw.get("agent"), "agent")

    @property
    def providers(self) -> dict[str, Any]:
        return required_mapping(self.raw.get("providers"), "providers")

    @property
    def name(self) -> str:
        return required_string(self.benchmark.get("name"), "benchmark.name")

    @property
    def scene_id(self) -> str:
        value = required_string(self.benchmark.get("scene_id"), "benchmark.scene_id")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ManifestError(
                "`benchmark.scene_id` may contain only letters, digits, '.', '_', and '-'"
            )
        return value

    def repository_path(
        self,
        value: Any,
        field: str,
        *,
        must_exist: bool = True,
    ) -> Path:
        relative = Path(required_string(value, field))
        if relative.is_absolute():
            raise ManifestError(f"`{field}` must be relative to the repository root")
        resolved = (ROOT / relative).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as error:
            raise ManifestError(
                f"`{field}` escapes the repository: {relative}"
            ) from error
        if must_exist and not resolved.is_file():
            raise ManifestError(f"`{field}` does not exist: {relative}")
        return resolved

    @property
    def scene_graph(self) -> Path:
        return self.repository_path(
            self.benchmark.get("scene_graph"), "benchmark.scene_graph"
        )

    @property
    def question_paths(self) -> dict[str, Path]:
        questions = required_mapping(
            self.benchmark.get("questions"), "benchmark.questions"
        )
        return {
            task: self.repository_path(
                questions.get(task), f"benchmark.questions.{task}"
            )
            for task in TASK_SPECS
        }

    @property
    def question_metadata(self) -> Path:
        questions = required_mapping(
            self.benchmark.get("questions"), "benchmark.questions"
        )
        return self.repository_path(
            questions.get("metadata"), "benchmark.questions.metadata"
        )

    @property
    def output_dir(self) -> Path:
        output = self.repository_path(
            self.benchmark.get("output_dir"),
            "benchmark.output_dir",
            must_exist=False,
        )
        output_root = (ROOT / "output").resolve()
        try:
            output.relative_to(output_root)
        except ValueError as error:
            raise ManifestError(
                "`benchmark.output_dir` must be below `output/`"
            ) from error
        return output

    def provider(self, name: str) -> dict[str, Any]:
        if name not in SUPPORTED_PROVIDERS:
            raise ManifestError(f"Unsupported provider: {name}")
        return required_mapping(self.providers.get(name), f"providers.{name}")

    def enabled_models(self, provider: str) -> list[dict[str, Any]]:
        config = self.provider(provider)
        models = config.get("models")
        if not isinstance(models, list):
            raise ManifestError(f"`providers.{provider}.models` must be a list")
        aliases: set[str] = set()
        enabled = []
        for index, model in enumerate(models):
            model = required_mapping(model, f"providers.{provider}.models[{index}]")
            alias = required_string(
                model.get("alias"), f"providers.{provider}.models[{index}].alias"
            )
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
                raise ManifestError(
                    f"Model alias `{alias}` may contain only letters, digits, '.', "
                    "'_', and '-'"
                )
            required_string(
                model.get("model"), f"providers.{provider}.models[{index}].model"
            )
            enabled_value = model.get("enabled", True)
            if not isinstance(enabled_value, bool):
                raise ManifestError(
                    f"`providers.{provider}.models[{index}].enabled` must be boolean"
                )
            if alias in aliases:
                raise ManifestError(
                    f"Duplicate model alias `{alias}` for provider `{provider}`"
                )
            aliases.add(alias)
            if enabled_value:
                enabled.append(deepcopy(model))
        return enabled

    def validate_question_metadata(self) -> None:
        scene_hash = file_sha256(self.scene_graph)
        metadata_path = self.question_metadata
        question_paths = self.question_paths
        question_directory = metadata_path.parent
        if question_directory.name != self.scene_id:
            raise ManifestError(
                "Question files must be grouped in a directory named after "
                "`benchmark.scene_id`"
            )
        if any(path.parent != question_directory for path in question_paths.values()):
            raise ManifestError(
                "QA, PDDL, and metadata files must share the scene question directory"
            )
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = required_mapping(
                yaml.safe_load(stream), "benchmark.questions.metadata contents"
            )
        if metadata.get("schema_version") != 1:
            raise ManifestError("Question metadata must use `schema_version: 1`")
        if metadata.get("scene_id") != self.scene_id:
            raise ManifestError(
                "Question metadata scene ID does not match `benchmark.scene_id`"
            )
        metadata_scene = required_mapping(
            metadata.get("scene_graph"), "question metadata scene_graph"
        )
        expected_scene_path = self.scene_graph.relative_to(ROOT).as_posix()
        if metadata_scene.get("path") != expected_scene_path:
            raise ManifestError(
                "Question metadata scene path does not match `benchmark.scene_graph`"
            )
        if metadata_scene.get("sha256") != scene_hash:
            raise ManifestError(
                "Question metadata was generated from a different scene graph checksum"
            )
        artifacts = required_mapping(
            metadata.get("artifacts"), "question metadata artifacts"
        )
        parameters = required_mapping(
            metadata.get("parameters"), "question metadata parameters"
        )
        for task, path in question_paths.items():
            artifact = required_mapping(
                artifacts.get(task), f"question metadata artifacts.{task}"
            )
            expected_question_path = path.relative_to(ROOT).as_posix()
            if artifact.get("path") != expected_question_path:
                raise ManifestError(
                    f"Question metadata path does not match the `{task}` manifest path"
                )
            if artifact.get("sha256") != file_sha256(path):
                raise ManifestError(
                    f"Question file checksum does not match metadata for `{task}`"
                )
            with path.open("r", encoding="utf-8") as stream:
                question_data = required_mapping(
                    yaml.safe_load(stream), f"{task} question file"
                )
            if "metadata" in question_data:
                raise ManifestError(
                    f"Question file `{task}` must keep metadata only in metadata.yaml"
                )
            questions = question_data.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ManifestError(f"Question file `{task}` contains no questions")
            expected_count = parameters.get(f"{task}_question_count")
            if expected_count != len(questions):
                raise ManifestError(
                    f"Question count does not match metadata for `{task}`"
                )
            if len(questions) != 50:
                raise ManifestError(
                    f"Benchmark question file `{task}` must contain exactly 50 questions"
                )

    def validate(self, selected_providers: Sequence[str] = ()) -> None:
        if self.raw.get("schema_version") != 1:
            raise ManifestError(
                "Only benchmark manifest `schema_version: 1` is supported"
            )
        _ = (
            self.name,
            self.scene_id,
            self.scene_graph,
            self.question_paths,
            self.question_metadata,
            self.output_dir,
        )
        self.validate_question_metadata()

        temperature = self.agent.get("temperature")
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ManifestError("`agent.temperature` must be a number")
        seed = self.agent.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestError("`agent.seed` must be an integer")
        max_iterations = self.agent.get("max_iterations")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ManifestError("`agent.max_iterations` must be an integer")
        if max_iterations < 1:
            raise ManifestError("`agent.max_iterations` must be at least one")

        for provider in SUPPORTED_PROVIDERS:
            config = self.provider(provider)
            if not isinstance(config.get("enabled"), bool):
                raise ManifestError(f"`providers.{provider}.enabled` must be boolean")
            self.enabled_models(provider)
            local_metrics = config.get("local_metrics")
            if local_metrics is not None:
                local_metrics = required_mapping(
                    local_metrics, f"providers.{provider}.local_metrics"
                )
                warmup_enabled = local_metrics.get("warmup_enabled", True)
                if not isinstance(warmup_enabled, bool):
                    raise ManifestError(
                        f"`providers.{provider}.local_metrics.warmup_enabled` "
                        "must be boolean"
                    )
                warmup_requests = local_metrics.get("warmup_requests", 1)
                if (
                    isinstance(warmup_requests, bool)
                    or not isinstance(warmup_requests, int)
                    or warmup_requests < 1
                ):
                    raise ManifestError(
                        f"`providers.{provider}.local_metrics.warmup_requests` "
                        "must be a positive integer"
                    )
                if warmup_enabled:
                    required_string(
                        local_metrics.get("warmup_prompt", "Reply with OK."),
                        f"providers.{provider}.local_metrics.warmup_prompt",
                    )
        for provider in selected_providers:
            config = self.provider(provider)
            if not config["enabled"]:
                raise ManifestError(
                    f"Provider `{provider}` is disabled in the benchmark manifest"
                )
            if not self.enabled_models(provider):
                raise ManifestError(
                    f"Provider `{provider}` has no enabled models in the manifest"
                )

    def experiment(self, provider: str, task: str) -> dict[str, Any]:
        spec = TASK_SPECS[task]
        provider_config = self.provider(provider)
        prompt_settings: dict[str, Any] = {
            "base_prompt": spec["prompt"],
            "output_type": spec["output_type"],
        }
        if task == "qa":
            prompt_settings["answer_type_hint"] = True

        metadata: dict[str, Any] = {
            "benchmark_family": f"{provider}_model_sweep",
            "benchmark_name": self.name,
            "benchmark_manifest": self.path.relative_to(ROOT).as_posix(),
            "scene_id": self.scene_id,
            "scene_graph_sha256": file_sha256(self.scene_graph),
            "question_metadata": self.question_metadata.relative_to(ROOT).as_posix(),
            "task": spec["task"],
        }
        local_metrics = provider_config.get("local_metrics")
        if local_metrics is not None:
            metadata["local_metrics"] = required_mapping(
                local_metrics, f"providers.{provider}.local_metrics"
            )

        return {
            "metadata": metadata,
            "model_sweeps": {
                f"{provider}-{spec['name']}": {
                    "provider": provider,
                    "phase": "main",
                    "configuration_name_template": spec["configuration"],
                    "result_configuration_name": spec["result_configuration"],
                    "models": deepcopy(provider_config["models"]),
                    "template": {
                        "dsg_interface": {"dsg_interface_type": "none"},
                        "pipeline": "agentic",
                        "phases": {
                            "main": {
                                "client": {"client_type": provider},
                                "model_info": {
                                    "temperature": self.agent["temperature"],
                                    "seed": self.agent["seed"],
                                },
                                "agent_info": {
                                    "prompt_settings": prompt_settings,
                                    "tools": [
                                        {
                                            "name": "run_cypher_query",
                                            "bound_args": {
                                                "dsgdb_conf": {
                                                    "dsg_interface_type": "heracles",
                                                    "uri": "$HERACLES_NEO4J_URI",
                                                }
                                            },
                                        }
                                    ],
                                    "tool_interface": provider,
                                    "max_iterations": self.agent["max_iterations"],
                                },
                            }
                        },
                        "questions": str(self.question_paths[task]),
                    },
                }
            },
            "configurations": {},
        }

    def prepare(self, runtime_dir: Path, providers: Sequence[str]) -> list[Path]:
        self.validate(providers)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for provider in providers:
            provider_dir = runtime_dir / provider
            provider_dir.mkdir(parents=True, exist_ok=True)
            for task, spec in TASK_SPECS.items():
                path = provider_dir / spec["filename"]
                with path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(
                        self.experiment(provider, task), stream, sort_keys=False
                    )
                paths.append(path)

        with self.question_metadata.open("r", encoding="utf-8") as stream:
            question_generation = required_mapping(
                yaml.safe_load(stream), "benchmark.questions.metadata contents"
            )

        snapshot = {
            "schema_version": 1,
            "source_manifest": self.path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": file_sha256(self.path),
            "selected_providers": list(providers),
            "input_sha256": {
                "scene_graph": file_sha256(self.scene_graph),
                "qa_questions": file_sha256(self.question_paths["qa"]),
                "pddl_questions": file_sha256(self.question_paths["pddl"]),
                "question_metadata": file_sha256(self.question_metadata),
            },
            "question_generation": question_generation,
            "resolved_manifest": self.raw,
        }
        with (runtime_dir / "benchmark_manifest.resolved.yaml").open(
            "w", encoding="utf-8"
        ) as stream:
            yaml.safe_dump(snapshot, stream, sort_keys=False)
        return paths


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise ManifestError(f"Ollama request failed for {url}: {error}") from error
    return required_mapping(result, f"response from {url}")


def pull_ollama_models(manifest: BenchmarkManifest, host: str) -> None:
    manifest.validate(("ollama",))
    host = host.rstrip("/")
    tags = request_json(f"{host}/api/tags")
    installed = {
        str(item.get("name"))
        for item in tags.get("models", [])
        if isinstance(item, dict) and item.get("name")
    }
    for entry in manifest.enabled_models("ollama"):
        model = str(entry["model"])
        if model in installed:
            print(f"Ollama model already available: {model}", flush=True)
            continue
        print(f"Pulling missing Ollama model: {model}", flush=True)
        request_json(f"{host}/api/pull", payload={"name": model, "stream": False})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "benchmark.yaml",
        help="Benchmark manifest (default: configs/benchmark.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--provider", action="append", choices=SUPPORTED_PROVIDERS, default=[]
    )

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--runtime-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--provider", action="append", choices=SUPPORTED_PROVIDERS, required=True
    )

    value_parser = subparsers.add_parser("value")
    value_parser.add_argument(
        "field", choices=("name", "scene_id", "scene_graph", "output_dir")
    )

    models_parser = subparsers.add_parser("models")
    models_parser.add_argument("provider", choices=SUPPORTED_PROVIDERS)

    pull_parser = subparsers.add_parser("pull-ollama")
    pull_parser.add_argument(
        "--host", default=os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = BenchmarkManifest.load(args.config)
        if args.command == "validate":
            manifest.validate(args.provider)
            print(f"Valid benchmark manifest: {manifest.path}")
        elif args.command == "prepare":
            paths = manifest.prepare(args.runtime_dir, args.provider)
            for path in paths:
                print(path)
        elif args.command == "value":
            manifest.validate()
            values = {
                "name": manifest.name,
                "scene_id": manifest.scene_id,
                "scene_graph": str(manifest.scene_graph),
                "output_dir": str(manifest.output_dir),
            }
            print(values[args.field])
        elif args.command == "models":
            manifest.validate((args.provider,))
            for model in manifest.enabled_models(args.provider):
                print(model["model"])
        elif args.command == "pull-ollama":
            pull_ollama_models(manifest, args.host)
        return 0
    except (ManifestError, OSError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect provider metadata and propose benchmark model capabilities."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


OPENROUTER_REASONING_EFFORTS = [
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
]
OPENROUTER_REASONING_DOCS = (
    "https://openrouter.ai/docs/guides/best-practices/reasoning-tokens"
)
OPENROUTER_MODELS_API = "https://openrouter.ai/api/v1/models"
OLLAMA_THINKING_DOCS = "https://docs.ollama.com/capabilities/thinking"
OLLAMA_SHOW_DOCS = "https://docs.ollama.com/api-reference/show-model-details"
ROOT = Path(__file__).resolve().parents[1]
OLLAMA_COMPOSE_FILE = ROOT / "docker" / "benchmark" / "docker-compose.yaml"


class InspectionError(RuntimeError):
    """Provider metadata could not be retrieved or interpreted."""


class ProviderConnectionError(InspectionError):
    """The provider API could not be reached."""


@dataclass(frozen=True)
class JsonResponse:
    payload: dict[str, Any]
    url: str


JsonFetcher = Callable[[str, dict[str, str], float, bytes | None], JsonResponse]


def fetch_json(
    url: str,
    headers: dict[str, str],
    timeout: float,
    body: bytes | None = None,
) -> JsonResponse:
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            final_url = response.geturl()
    except urllib.error.HTTPError as error:
        raise InspectionError(
            f"Request failed for {url}: HTTP {error.code} {error.reason}"
        ) from error
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        OSError,
        http.client.HTTPException,
    ) as error:
        raise ProviderConnectionError(f"Request failed for {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise InspectionError(f"Request failed for {url}: {error}") from error
    if not isinstance(payload, dict):
        raise InspectionError(f"Expected a JSON object from {url}")
    return JsonResponse(payload=payload, url=final_url)


def inspect_openrouter(
    model: str,
    *,
    timeout: float = 15,
    fetcher: JsonFetcher = fetch_json,
) -> dict[str, Any]:
    encoded_model = urllib.parse.quote(model, safe="/")
    url = f"https://openrouter.ai/api/v1/model/{encoded_model}"
    headers = {"Accept": "application/json", "User-Agent": "heracles-benchmark"}
    api_key = os.environ.get("HERACLES_OPENROUTER_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = fetcher(url, headers, timeout, None)
    data = response.payload.get("data")
    if not isinstance(data, dict):
        raise InspectionError(
            f"OpenRouter did not return model metadata for `{model}`. Check the "
            f"identifier in {OPENROUTER_MODELS_API}."
        )

    supported_parameters = data.get("supported_parameters")
    if not isinstance(supported_parameters, list):
        supported_parameters = []
    supported_parameters = [str(value) for value in supported_parameters]
    reasoning = data.get("reasoning")
    missing: list[dict[str, str]] = []

    if isinstance(reasoning, dict):
        reasoning_support = "required" if reasoning.get("mandatory") is True else "optional"
        if "supported_efforts" not in reasoning:
            reasoning_efforts = None
        elif reasoning["supported_efforts"] is None:
            # OpenRouter null means that all gateway effort values are accepted.
            reasoning_efforts = list(OPENROUTER_REASONING_EFFORTS)
        elif isinstance(reasoning["supported_efforts"], list):
            reasoning_efforts = [
                str(value)
                for value in reasoning["supported_efforts"]
                if str(value) != "none"
            ]
        else:
            reasoning_efforts = None
            missing.append(
                missing_item(
                    "reasoning_efforts",
                    "The API returned an unrecognized supported_efforts value.",
                    OPENROUTER_REASONING_DOCS,
                )
            )
    elif "reasoning" in supported_parameters or "reasoning_effort" in supported_parameters:
        reasoning_support = "optional"
        reasoning_efforts = None
        missing.append(
            missing_item(
                "reasoning_efforts",
                "The model accepts reasoning parameters but exposes no reasoning metadata.",
                OPENROUTER_REASONING_DOCS,
            )
        )
    else:
        reasoning_support = "unsupported"
        reasoning_efforts = None

    capabilities = {
        "reasoning": reasoning_support,
        "reasoning_efforts": reasoning_efforts,
        "temperature": "temperature" in supported_parameters,
        "seed": "seed" in supported_parameters,
    }
    return inspection_result(
        provider="openrouter",
        model=model,
        capabilities=capabilities,
        evidence={
            "supported_parameters": supported_parameters,
            "reasoning": reasoning,
        },
        retrieved_from=response.url,
        missing=missing,
        references=[OPENROUTER_MODELS_API, OPENROUTER_REASONING_DOCS],
    )


def inspect_ollama(
    model: str,
    *,
    host: str = "http://localhost:11434",
    timeout: float = 15,
    fetcher: JsonFetcher = fetch_json,
) -> dict[str, Any]:
    url = f"{host.rstrip('/')}/api/show"
    body = json.dumps({"model": model}).encode("utf-8")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = fetcher(
            url,
            headers,
            timeout,
            body,
        )
    except InspectionError as error:
        if "HTTP 404" in str(error):
            raise InspectionError(
                f"Ollama model `{model}` is not installed. Install it with: "
                f"docker exec ollama ollama pull {model}"
            ) from error
        raise
    capabilities_value = response.payload.get("capabilities")
    provider_capabilities = (
        [str(value) for value in capabilities_value]
        if isinstance(capabilities_value, list)
        else []
    )
    model_family = model.split(":", 1)[0]
    library_url = f"https://ollama.com/library/{urllib.parse.quote(model_family)}"
    model_info = response.payload.get("model_info")
    model_info = model_info if isinstance(model_info, dict) else {}
    details = response.payload.get("details")
    details = details if isinstance(details, dict) else {}
    families = details.get("families", [])
    families = families if isinstance(families, list) else []
    searchable_identity = " ".join(
        [model, *[str(value) for value in families], *model_info.keys()]
    ).lower()
    supports_thinking = "thinking" in provider_capabilities
    is_gpt_oss = "gpt-oss" in searchable_identity or "gpt_oss" in searchable_identity
    missing: list[dict[str, str]] = []

    if not supports_thinking:
        reasoning_support = "unsupported"
        reasoning_efforts = None
    elif is_gpt_oss:
        reasoning_support = "required"
        reasoning_efforts = ["low", "medium", "high"]
    else:
        reasoning_support = "optional"
        reasoning_efforts = None
        missing.extend(
            [
                missing_item(
                    "reasoning requirement",
                    "Ollama /api/show reports thinking support but not whether it can be disabled.",
                    library_url,
                ),
                missing_item(
                    "reasoning_efforts",
                    "Ollama /api/show does not report model-specific thinking levels.",
                    OLLAMA_THINKING_DOCS,
                ),
            ]
        )

    # These are Ollama runtime options rather than entries in /api/show's
    # feature list. A model-specific rejection still needs a small dry-run.
    capabilities = {
        "reasoning": reasoning_support,
        "reasoning_efforts": reasoning_efforts,
        "temperature": True,
        "seed": True,
    }
    missing.append(
        missing_item(
            "temperature and seed model compatibility",
            "Ollama exposes these as runtime options, but /api/show does not "
            "declare support per model.",
            OLLAMA_SHOW_DOCS,
        )
    )
    return inspection_result(
        provider="ollama",
        model=model,
        capabilities=capabilities,
        evidence={
            "capabilities": provider_capabilities,
            "parameters": response.payload.get("parameters"),
            "families": families,
        },
        retrieved_from=response.url,
        missing=missing,
        references=[library_url, OLLAMA_SHOW_DOCS, OLLAMA_THINKING_DOCS],
    )


def is_local_ollama_host(host: str) -> bool:
    parsed = urllib.parse.urlparse(host)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def start_ollama_container(
    host: str,
    *,
    timeout: float = 60,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    readiness_fetcher: JsonFetcher = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Start the repository Ollama service and wait for its API."""

    if not OLLAMA_COMPOSE_FILE.is_file():
        raise InspectionError(
            f"Ollama Compose file does not exist: {OLLAMA_COMPOSE_FILE}"
        )
    environment = os.environ.copy()
    environment.setdefault("BENCHMARK_HOST_OUTPUT_DIR", str(ROOT / "output"))
    command = [
        "docker",
        "compose",
        "-f",
        str(OLLAMA_COMPOSE_FILE),
        "up",
        "-d",
        "ollama",
    ]
    try:
        completed = runner(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise InspectionError(
            "Docker is required to start the local Ollama service, but the "
            "docker command was not found."
        ) from error
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown error").strip()
        raise InspectionError(f"Could not start the Ollama container: {details}")

    deadline = time.monotonic() + timeout
    tags_url = f"{host.rstrip('/')}/api/tags"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            readiness_fetcher(tags_url, {"Accept": "application/json"}, 2, None)
            return
        except InspectionError as error:
            last_error = error
            sleep(1)
    raise InspectionError(
        f"Ollama container started, but its API was not ready after {timeout:g}s: "
        f"{last_error}"
    )


def missing_item(field: str, reason: str, reference: str) -> dict[str, str]:
    return {"field": field, "reason": reason, "reference": reference}


def inspection_result(
    *,
    provider: str,
    model: str,
    capabilities: dict[str, Any],
    evidence: dict[str, Any],
    retrieved_from: str,
    missing: list[dict[str, str]],
    references: list[str],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "retrieved_from": retrieved_from,
        "capabilities": capabilities,
        "missing_information": missing,
        "evidence": evidence,
        "references": references,
    }


def render_rich(result: dict[str, Any], console: Console) -> None:
    console.print(
        Panel.fit(
            f"[bold]{result['provider']}[/bold] / {result['model']}",
            title="Model capability inspection",
        )
    )
    console.print("\n[bold]Suggested benchmark.yaml block[/bold]")
    console.print(
        yaml.safe_dump(
            {"capabilities": result["capabilities"]},
            sort_keys=False,
        ).rstrip()
    )
    missing = result["missing_information"]
    if missing:
        table = Table(title="Information requiring confirmation", show_lines=True)
        table.add_column("Field", style="yellow")
        table.add_column("Why it is missing")
        table.add_column("Where to check", style="cyan")
        for item in missing:
            table.add_row(item["field"], item["reason"], item["reference"])
        console.print(table)
    else:
        console.print(
            "[green]No capability fields are missing from provider metadata.[/green]"
        )
    console.print(f"\n[dim]Retrieved from: {result['retrieved_from']}[/dim]")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("openrouter", "ollama"))
    parser.add_argument("model", help="Exact provider model identifier")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument(
        "--no-start-ollama",
        action="store_true",
        help="Do not start the repository Ollama container when localhost is offline",
    )
    parser.add_argument(
        "--ollama-start-timeout",
        type=float,
        default=60,
        help="Seconds to wait for an automatically started Ollama API",
    )
    parser.add_argument("--format", choices=("rich", "yaml", "json"), default="rich")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.provider == "openrouter":
            result = inspect_openrouter(args.model, timeout=args.timeout)
        else:
            try:
                result = inspect_ollama(
                    args.model,
                    host=args.ollama_host,
                    timeout=args.timeout,
                )
            except ProviderConnectionError:
                if args.no_start_ollama or not is_local_ollama_host(args.ollama_host):
                    raise
                Console(stderr=True).print(
                    "[yellow]Local Ollama is unavailable; starting the repository "
                    "container...[/yellow]"
                )
                start_ollama_container(
                    args.ollama_host,
                    timeout=args.ollama_start_timeout,
                )
                result = inspect_ollama(
                    args.model,
                    host=args.ollama_host,
                    timeout=args.timeout,
                )
    except InspectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2))
    elif args.format == "yaml":
        print(yaml.safe_dump(result, sort_keys=False).rstrip())
    else:
        render_rich(result, Console())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render quiet, structured progress for Heracles benchmark subprocesses."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO

import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console(highlight=False)
PROVIDER_LABELS = {"ollama": "Ollama", "openrouter": "OpenRouter"}
CONFIGURATION_RE = re.compile(r"Running configuration: (.+?)\s*$")
EXPERIMENT_RE = re.compile(r"Running experiment: (.+?)\s*$")
RESULT_RE = re.compile(r"Saved results: (.+?)\s*$")
WARMUP_RE = re.compile(r"Warming Ollama model: (.+?) \(\d+/\d+\)\s*$")
QUESTION_SWEEP_RE = re.compile(r"Question sweep size: (\d+)\s*$")
QUESTION_PROGRESS_RE = re.compile(
    r"Question progress: (?P<completed>\d+)/(?P<total>\d+) \| (?P<label>.+?)\s*$"
)
SCENE_COUNT_RE = re.compile(
    r"^# (?P<label>Objects|Places|2D Places|Rooms):\s+(?P<count>\d+)\s*$"
)


@dataclass
class Diagnostics:
    """Counts for noisy messages retained in the detailed log."""

    http_requests: int = 0
    database_notifications: int = 0
    validation_messages: int = 0
    query_errors: int = 0
    other_warnings: int = 0

    def observe(self, line: str) -> None:
        if "httpx:HTTP Request:" in line:
            self.http_requests += 1
        elif "neo4j.notifications:" in line:
            self.database_notifications += 1
        elif "pipelines.comparisons:" in line or " not in [Fact(" in line:
            self.validation_messages += 1
        elif line.lstrip().startswith("{neo4j_code:"):
            self.query_errors += 1
        elif "WARNING:" in line:
            self.other_warnings += 1

    def rows(self) -> list[tuple[str, int]]:
        labels = {
            "http_requests": "HTTP request logs",
            "database_notifications": "Neo4j notifications",
            "validation_messages": "Answer-validation messages",
            "query_errors": "Tool query errors",
            "other_warnings": "Other warnings",
        }
        return [
            (labels[field.name], getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name)
        ]


@dataclass(frozen=True)
class PlannedConfiguration:
    name: str
    task: str
    model: str


@dataclass(frozen=True)
class ResultRow:
    task: str
    model: str
    questions: int
    final_answer_match: int
    cypher_solution_match: int
    cypher_solution_evaluated: int
    tool_executable: int
    tool_executable_evaluated: int
    throughput: float | None


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def normalize_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if normalized[:1] == ["--"]:
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("A command is required after `--`")
    return normalized


def open_log(path: Path, *, reset: bool = False) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if reset else "a"
    stream = path.open(mode, encoding="utf-8", buffering=1)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return stream


def run_streamed(
    command: Sequence[str],
    log_path: Path,
    *,
    label: str,
    reset_log: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> tuple[int, float, list[str]]:
    """Run a child quietly, retaining its complete output and a short tail."""

    started = perf_counter()
    tail: deque[str] = deque(maxlen=12)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open_log(log_path, reset=reset_log) as log:
        log.write(f"\n===== {label} =====\n")
        log.write(f"$ {shlex.join(command)}\n")
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                stripped = line.rstrip()
                if stripped:
                    tail.append(stripped)
                if on_line is not None:
                    on_line(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
            log.write(f"===== exit code: {process.returncode} =====\n")
    return return_code, perf_counter() - started, list(tail)


def failure_panel(label: str, log_path: Path, tail: Sequence[str]) -> None:
    excerpt = "\n".join(tail[-8:]) or "No diagnostic output was produced."
    console.print(
        Panel(
            excerpt,
            title=f"[bold red]{label} failed[/bold red]",
            subtitle=f"Details: {display_path(log_path)}",
            border_style="red",
        )
    )


def summary_table(title: str, rows: Sequence[tuple[str, Any]]) -> None:
    table = Table(
        title=title,
        title_style="bold cyan",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


def run_quiet_command(args: argparse.Namespace) -> int:
    command = normalize_command(args.command)
    captured_lines: list[str] = []
    with console.status(f"[bold cyan]{args.label}[/bold cyan]", spinner="dots"):
        return_code, elapsed, tail = run_streamed(
            command,
            args.log_file,
            label=args.label,
            reset_log=args.reset_log,
            on_line=captured_lines.append,
        )
    if return_code:
        failure_panel(args.label, args.log_file, tail)
        return return_code

    console.print(f"[green]✓[/green] {args.label} [dim]({elapsed:.1f}s)[/dim]")
    if args.kind == "scene":
        counts = {}
        for line in captured_lines:
            match = SCENE_COUNT_RE.match(line.strip())
            if match:
                counts[match.group("label")] = match.group("count")
        if counts:
            summary_table(
                "Scene graph",
                [
                    ("Objects", counts.get("Objects", "-")),
                    ("Places", counts.get("Places", "-")),
                    ("2D places", counts.get("2D Places", "-")),
                    ("Rooms", counts.get("Rooms", "-")),
                ],
            )
    if args.success_detail:
        console.print(f"  [dim]{args.success_detail}[/dim]")
    return 0


def task_name(data: dict[str, Any], path: Path) -> str:
    metadata = data.get("metadata")
    raw_task = metadata.get("task") if isinstance(metadata, dict) else None
    if raw_task == "pddl" or "pddl" in path.stem.casefold():
        return "PDDL"
    return "QA"


def experiment_plan(paths: Sequence[Path]) -> list[PlannedConfiguration]:
    plan = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or not isinstance(data.get("model_sweeps"), dict):
            raise TypeError(f"Experiment has no model sweep: {path}")
        task = task_name(data, path)
        for sweep_name, sweep in data["model_sweeps"].items():
            if not isinstance(sweep, dict):
                raise TypeError(f"Invalid model sweep `{sweep_name}` in {path}")
            models = sweep.get("models")
            template = sweep.get("configuration_name_template")
            if not isinstance(models, list) or not isinstance(template, str):
                raise TypeError(f"Incomplete model sweep `{sweep_name}` in {path}")
            for model in models:
                if not isinstance(model, dict) or model.get("enabled", True) is False:
                    continue
                alias = model.get("alias")
                identifier = model.get("model")
                if not isinstance(alias, str) or not isinstance(identifier, str):
                    raise TypeError(f"Invalid model entry in {path}")
                plan.append(
                    PlannedConfiguration(
                        name=template.format(alias=alias, model=identifier),
                        task=task,
                        model=identifier,
                    )
                )
    return plan


def result_row(path: Path) -> ResultRow | None:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata", {})
    configurations = data.get("experiment_configurations", {})
    if not isinstance(metadata, dict) or not isinstance(configurations, dict):
        return None
    configuration = next(iter(configurations.values()), None)
    if not isinstance(configuration, dict):
        return None
    summary = configuration.get("analysis_summary")
    if not isinstance(summary, dict):
        return None
    llm_configurations = metadata.get("llm_configurations", {})
    model = "-"
    if isinstance(llm_configurations, dict):
        llm_configuration = next(iter(llm_configurations.values()), None)
        if isinstance(llm_configuration, dict):
            phases = llm_configuration.get("phases", {})
            if isinstance(phases, dict):
                phase = next(iter(phases.values()), None)
                if isinstance(phase, dict) and isinstance(
                    phase.get("model_identifier"), str
                ):
                    model = phase["model_identifier"]
    task = "PDDL" if metadata.get("task") == "pddl" else "QA"
    return ResultRow(
        task=task,
        model=model,
        questions=int(summary.get("questions", 0)),
        final_answer_match=int(
            summary.get("final_answer_match_count", summary.get("correct_count", 0))
        ),
        cypher_solution_match=int(summary.get("cypher_solution_match_count", 0)),
        cypher_solution_evaluated=int(
            summary.get("cypher_solution_match_evaluated", 0)
        ),
        tool_executable=int(summary.get("tool_executable_count", 0)),
        tool_executable_evaluated=int(summary.get("tool_executable_evaluated", 0)),
        throughput=summary.get("output_tokens_per_second")
        if isinstance(summary.get("output_tokens_per_second"), int | float)
        else None,
    )


def print_result_rows(provider: str, paths: Sequence[Path]) -> None:
    rows = []
    for path in paths:
        try:
            row = result_row(path)
        except (OSError, ValueError, yaml.YAMLError):
            row = None
        if row is not None:
            rows.append(row)
    if not rows:
        return
    table = Table(
        title=f"{PROVIDER_LABELS.get(provider, provider)} results",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Task")
    table.add_column("Model", overflow="fold")
    table.add_column("Questions", justify="right")
    table.add_column("Tool Executable", justify="right")
    table.add_column("Cypher / Grounding", justify="right")
    table.add_column("Final Answer", justify="right")
    table.add_column("Throughput", justify="right")
    for row in sorted(rows, key=lambda item: (item.task, item.model)):
        throughput = "-" if row.throughput is None else f"{row.throughput:.2f} tok/s"
        table.add_row(
            row.task,
            row.model,
            str(row.questions),
            f"{row.tool_executable}/{row.tool_executable_evaluated}",
            f"{row.cypher_solution_match}/{row.cypher_solution_evaluated}",
            f"{row.final_answer_match}/{row.questions}",
            throughput,
        )
    console.print(table)


def print_diagnostics(diagnostics: Diagnostics, log_path: Path) -> None:
    rows = diagnostics.rows()
    if rows:
        rows.append(("Detailed log", display_path(log_path)))
        summary_table("Diagnostics captured", rows)
    else:
        console.print(f"  [dim]Detailed log: {display_path(log_path)}[/dim]")


def run_experiments(args: argparse.Namespace) -> int:
    paths = [path.expanduser().resolve() for path in args.experiment]
    plan = experiment_plan(paths)
    if not plan:
        raise ValueError("No enabled model configurations were found")
    by_name = {item.name: item for item in plan}
    diagnostics = Diagnostics()
    provider_label = PROVIDER_LABELS.get(args.provider, args.provider)
    result_paths: list[Path] = []
    current_name: str | None = None
    current_task = "Preparing"
    question_progress_id: int | None = None
    completed: set[tuple[str, str]] = set()

    command = [
        sys.executable,
        "-u",
        str(args.runner.expanduser().resolve()),
        *(str(path) for path in paths),
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
        "--no-display",
        "--log-level",
        "INFO",
    ]

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[current]}"),
        console=console,
    ) as progress:
        progress_id = progress.add_task(
            f"{provider_label} sweep",
            total=len(plan),
            current="Preparing",
        )

        def complete_current() -> None:
            nonlocal current_name, question_progress_id
            if current_name is None:
                return
            key = (current_task, current_name)
            if key not in completed:
                completed.add(key)
                progress.advance(progress_id)
            current_name = None
            if question_progress_id is not None:
                progress.remove_task(question_progress_id)
                question_progress_id = None

        def observe(line: str) -> None:
            nonlocal current_name, current_task, question_progress_id
            diagnostics.observe(line)
            experiment_match = EXPERIMENT_RE.search(line)
            if experiment_match:
                complete_current()
                experiment_path = Path(experiment_match.group(1))
                current_task = (
                    "PDDL" if "pddl" in experiment_path.stem.casefold() else "QA"
                )
            configuration_match = CONFIGURATION_RE.search(line)
            if configuration_match:
                complete_current()
                current_name = configuration_match.group(1)
                configuration = by_name.get(current_name)
                model = configuration.model if configuration else current_name
                progress.update(
                    progress_id,
                    current=f"{current_task} · {model}",
                )
            question_sweep_match = QUESTION_SWEEP_RE.search(line)
            if question_sweep_match:
                if question_progress_id is not None:
                    progress.remove_task(question_progress_id)
                question_progress_id = progress.add_task(
                    f"↳ {current_task} questions",
                    total=int(question_sweep_match.group(1)),
                    completed=0,
                    current="Starting",
                )
            question_match = QUESTION_PROGRESS_RE.search(line)
            if question_match:
                total = int(question_match.group("total"))
                if question_progress_id is None:
                    question_progress_id = progress.add_task(
                        f"↳ {current_task} questions",
                        total=total,
                        completed=0,
                        current="Starting",
                    )
                progress.update(
                    question_progress_id,
                    total=total,
                    completed=int(question_match.group("completed")),
                    current=question_match.group("label"),
                )
            warmup_match = WARMUP_RE.search(line)
            if warmup_match:
                progress.update(
                    progress_id,
                    current=f"{current_task} · {warmup_match.group(1)} · warmup",
                )
            result_match = RESULT_RE.search(line)
            if result_match:
                result_paths.append(Path(result_match.group(1)).expanduser())

        return_code, elapsed, tail = run_streamed(
            command,
            args.log_file,
            label=f"{provider_label} model sweep",
            on_line=observe,
        )
        if return_code == 0:
            complete_current()
        progress.update(
            progress_id,
            current="Complete" if return_code == 0 else "Failed",
        )

    if return_code:
        failure_panel(f"{provider_label} model sweep", args.log_file, tail)
        print_diagnostics(diagnostics, args.log_file)
        return return_code

    console.print(
        f"[green]✓[/green] {provider_label} sweep completed "
        f"[dim]({len(completed)} configurations, {elapsed:.1f}s)[/dim]"
    )
    print_result_rows(args.provider, result_paths)
    print_diagnostics(diagnostics, args.log_file)
    return 0


def print_summary_command(args: argparse.Namespace) -> int:
    rows = []
    for item in args.field:
        if "=" not in item:
            raise ValueError("Summary fields must use LABEL=VALUE")
        label, value = item.split("=", 1)
        rows.append((label, value))
    summary_table(args.title, rows)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run one quiet benchmark phase")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--log-file", required=True, type=Path)
    run_parser.add_argument("--reset-log", action="store_true")
    run_parser.add_argument("--kind", choices=("generic", "scene"), default="generic")
    run_parser.add_argument("--success-detail")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(function=run_quiet_command)

    experiment_parser = subparsers.add_parser(
        "experiments", help="Run model sweeps with structured progress"
    )
    experiment_parser.add_argument("--provider", required=True)
    experiment_parser.add_argument("--runner", required=True, type=Path)
    experiment_parser.add_argument("--output-dir", required=True, type=Path)
    experiment_parser.add_argument("--log-file", required=True, type=Path)
    experiment_parser.add_argument(
        "--experiment", required=True, action="append", type=Path
    )
    experiment_parser.set_defaults(function=run_experiments)

    summary_parser = subparsers.add_parser("summary", help="Print a summary table")
    summary_parser.add_argument("--title", required=True)
    summary_parser.add_argument("--field", required=True, action="append")
    summary_parser.set_defaults(function=print_summary_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return args.function(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted.[/yellow]")
        return 130
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

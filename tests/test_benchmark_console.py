from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_console


class DiagnosticsTest(unittest.TestCase):
    def test_noise_categories_are_counted(self) -> None:
        diagnostics = benchmark_console.Diagnostics()
        lines = [
            'INFO:httpx:HTTP Request: POST https://example.test "HTTP/1.1 200 OK"',
            "WARNING:neo4j.notifications:Received notification from DBMS server",
            "WARNING:heracles_agents.pipelines.comparisons:Invalid SLDP",
            "(visited-room R1) not in [Fact(head='visited-region', params=['R1'])]",
            "{neo4j_code: Neo.ClientError.Statement.SyntaxError}",
            "WARNING:example:another warning",
        ]
        for line in lines:
            diagnostics.observe(line)

        self.assertEqual(diagnostics.http_requests, 1)
        self.assertEqual(diagnostics.database_notifications, 1)
        self.assertEqual(diagnostics.validation_messages, 2)
        self.assertEqual(diagnostics.query_errors, 1)
        self.assertEqual(diagnostics.other_warnings, 1)


class ExperimentParsingTest(unittest.TestCase):
    def test_experiment_plan_uses_only_enabled_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pddl_model_sweep.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "metadata": {"task": "pddl"},
                        "model_sweeps": {
                            "example": {
                                "configuration_name_template": "config-{alias}",
                                "models": [
                                    {
                                        "alias": "enabled",
                                        "model": "provider/enabled",
                                        "enabled": True,
                                    },
                                    {
                                        "alias": "disabled",
                                        "model": "provider/disabled",
                                        "enabled": False,
                                    },
                                ],
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                benchmark_console.experiment_plan([path]),
                [
                    benchmark_console.PlannedConfiguration(
                        name="config-enabled",
                        task="PDDL",
                        model="provider/enabled",
                    )
                ],
            )

    def test_result_row_extracts_compact_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "metadata": {
                            "task": "cypher",
                            "llm_configurations": {
                                "qa": {
                                    "phases": {
                                        "main": {"model_identifier": "provider/model"}
                                    }
                                }
                            },
                        },
                        "experiment_configurations": {
                            "qa": {
                                "analysis_summary": {
                                    "questions": 50,
                                    "final_answer_match_count": 42,
                                    "cypher_solution_match_count": 40,
                                    "cypher_solution_match_evaluated": 45,
                                    "tool_executable_count": 44,
                                    "tool_executable_evaluated": 45,
                                    "output_tokens_per_second": 19.25,
                                }
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                benchmark_console.result_row(path),
                benchmark_console.ResultRow(
                    task="QA",
                    model="provider/model",
                    questions=50,
                    final_answer_match=42,
                    cypher_solution_match=40,
                    cypher_solution_evaluated=45,
                    tool_executable=44,
                    tool_executable_evaluated=45,
                    throughput=19.25,
                ),
            )

    def test_experiment_progress_completes_enabled_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "cypher_model_sweep.yaml"
            experiment.write_text(
                yaml.safe_dump(
                    {
                        "metadata": {"task": "cypher"},
                        "model_sweeps": {
                            "example": {
                                "configuration_name_template": "config-{alias}",
                                "models": [
                                    {
                                        "alias": "model",
                                        "model": "provider/model",
                                        "enabled": True,
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            arguments = SimpleNamespace(
                experiment=[experiment],
                provider="openrouter",
                runner=root / "runner.py",
                output_dir=root / "output",
                log_file=root / "benchmark.log",
            )

            def fake_run_streamed(*_args, on_line, **_kwargs):
                on_line(f"INFO:runner:Running experiment: {experiment}\n")
                on_line("INFO:runner:Running configuration: config-model\n")
                on_line("INFO:runner:Question sweep size: 2\n")
                on_line("INFO:pipeline:Question progress: 1/2 | qa-001\n")
                on_line("INFO:pipeline:Question progress: 2/2 | qa-002\n")
                return 0, 1.25, []

            with (
                patch.object(
                    benchmark_console,
                    "run_streamed",
                    side_effect=fake_run_streamed,
                ),
                patch.object(benchmark_console, "print_result_rows"),
                patch.object(benchmark_console, "print_diagnostics"),
            ):
                self.assertEqual(benchmark_console.run_experiments(arguments), 0)


class SubprocessCaptureTest(unittest.TestCase):
    def test_streamed_command_is_written_to_detailed_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "benchmark.log"
            return_code, _elapsed, tail = benchmark_console.run_streamed(
                [sys.executable, "-c", "print('hidden detail')"],
                log_path,
                label="Example phase",
                reset_log=True,
            )

            self.assertEqual(return_code, 0)
            self.assertEqual(tail, ["hidden detail"])
            self.assertIn("hidden detail", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

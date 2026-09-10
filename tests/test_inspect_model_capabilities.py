import subprocess
import unittest
from unittest.mock import patch

from scripts.inspect_model_capabilities import (
    InspectionError,
    JsonResponse,
    ProviderConnectionError,
    fetch_json,
    inspect_ollama,
    inspect_openrouter,
    is_local_ollama_host,
    main,
    start_ollama_container,
)


def fake_fetch(payload, final_url="https://provider.test/model"):
    def fetcher(url, headers, timeout, body):
        return JsonResponse(payload=payload, url=final_url)

    return fetcher


class OpenRouterInspectionTest(unittest.TestCase):
    def test_extracts_complete_capabilities(self):
        result = inspect_openrouter(
            "openai/gpt-oss-120b",
            fetcher=fake_fetch(
                {
                    "data": {
                        "supported_parameters": [
                            "reasoning",
                            "temperature",
                            "seed",
                        ],
                        "reasoning": {
                            "mandatory": True,
                            "supported_efforts": ["high", "medium", "low"],
                        },
                    }
                }
            ),
        )

        self.assertEqual(
            result["capabilities"],
            {
                "reasoning": "required",
                "reasoning_efforts": ["high", "medium", "low"],
                "temperature": True,
                "seed": True,
            },
        )
        self.assertEqual(result["missing_information"], [])

    def test_reports_missing_reasoning_metadata(self):
        result = inspect_openrouter(
            "provider/model",
            fetcher=fake_fetch(
                {"data": {"supported_parameters": ["reasoning", "temperature"]}}
            ),
        )

        self.assertEqual(result["capabilities"]["reasoning"], "optional")
        self.assertIsNone(result["capabilities"]["reasoning_efforts"])
        self.assertEqual(
            result["missing_information"][0]["field"], "reasoning_efforts"
        )


class OllamaInspectionTest(unittest.TestCase):
    def test_recognizes_mandatory_gpt_oss_thinking(self):
        result = inspect_ollama(
            "gpt-oss:20b",
            fetcher=fake_fetch(
                {
                    "capabilities": ["completion", "thinking", "tools"],
                    "details": {"families": ["gptoss"]},
                    "model_info": {},
                }
            ),
        )

        self.assertEqual(result["capabilities"]["reasoning"], "required")
        self.assertEqual(
            result["capabilities"]["reasoning_efforts"],
            ["low", "medium", "high"],
        )

    def test_points_to_model_docs_when_efforts_are_not_reported(self):
        result = inspect_ollama(
            "gemma4:26b",
            fetcher=fake_fetch(
                {
                    "capabilities": ["completion", "thinking"],
                    "details": {"families": ["gemma4"]},
                    "model_info": {},
                }
            ),
        )

        self.assertEqual(result["capabilities"]["reasoning"], "optional")
        self.assertIsNone(result["capabilities"]["reasoning_efforts"])
        fields = {item["field"] for item in result["missing_information"]}
        self.assertIn("reasoning requirement", fields)
        self.assertIn("reasoning_efforts", fields)
        self.assertIn("temperature and seed model compatibility", fields)

    def test_identifies_only_loopback_hosts_as_local(self):
        self.assertTrue(is_local_ollama_host("http://localhost:11434"))
        self.assertTrue(is_local_ollama_host("http://127.0.0.1:11434"))
        self.assertFalse(is_local_ollama_host("http://ollama:11434"))
        self.assertFalse(is_local_ollama_host("https://example.com"))

    @patch("scripts.inspect_model_capabilities.urllib.request.urlopen")
    def test_connection_reset_is_treated_as_retryable(self, urlopen_mock):
        urlopen_mock.side_effect = ConnectionResetError(104, "Connection reset")

        with self.assertRaises(ProviderConnectionError):
            fetch_json("http://localhost:11434/api/tags", {}, 2, None)

    def test_missing_model_error_includes_pull_command(self):
        def missing_fetcher(url, headers, timeout, body):
            raise InspectionError("HTTP 404 Not Found")

        with self.assertRaisesRegex(
            InspectionError,
            "docker exec ollama ollama pull gemma4:26b",
        ):
            inspect_ollama("gemma4:26b", fetcher=missing_fetcher)

    def test_starts_repository_compose_service_and_waits_until_ready(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "", "")

        readiness_calls = []

        def readiness_fetcher(url, headers, timeout, body):
            readiness_calls.append(url)
            if len(readiness_calls) == 1:
                raise ProviderConnectionError("not ready")
            return JsonResponse({"models": []}, url)

        start_ollama_container(
            "http://localhost:11434",
            runner=runner,
            readiness_fetcher=readiness_fetcher,
            sleep=lambda _seconds: None,
        )

        self.assertEqual(calls[0][0][-3:], ["up", "-d", "ollama"])
        self.assertIn("BENCHMARK_HOST_OUTPUT_DIR", calls[0][1]["env"])
        self.assertEqual(len(readiness_calls), 2)

    @patch("scripts.inspect_model_capabilities.render_rich")
    @patch("scripts.inspect_model_capabilities.start_ollama_container")
    @patch("scripts.inspect_model_capabilities.inspect_ollama")
    def test_main_starts_local_ollama_and_retries(
        self, inspect_mock, start_mock, render_mock
    ):
        result = {"provider": "ollama", "model": "gemma4:26b"}
        inspect_mock.side_effect = [ProviderConnectionError("offline"), result]

        exit_code = main(["ollama", "gemma4:26b"])

        self.assertEqual(exit_code, 0)
        start_mock.assert_called_once_with("http://localhost:11434", timeout=60)
        self.assertEqual(inspect_mock.call_count, 2)
        render_mock.assert_called_once()

    @patch("scripts.inspect_model_capabilities.inspect_ollama")
    def test_main_does_not_start_remote_ollama(self, inspect_mock):
        inspect_mock.side_effect = ProviderConnectionError("offline")

        with patch("builtins.print"):
            exit_code = main(
                ["ollama", "gemma4:26b", "--ollama-host", "http://ollama:11434"]
            )

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
from pathlib import Path
import unittest

from scripts.benchmark_manifest import BenchmarkManifest, ManifestError


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkInferenceParametersTest(unittest.TestCase):
    def load_manifest(self, filename: str = "benchmark.yaml") -> BenchmarkManifest:
        return BenchmarkManifest.load(ROOT / "configs" / filename)

    def test_default_and_smoke_manifests_have_normalized_reasoning(self) -> None:
        for filename in ("benchmark.yaml", "benchmark-smoke.yaml"):
            manifest = self.load_manifest(filename)
            manifest.validate()

            parameters = manifest.inference_parameters
            self.assertEqual(parameters["temperature"], manifest.agent["temperature"])
            self.assertEqual(parameters["seed"], manifest.agent["seed"])
            self.assertEqual(
                parameters["reasoning"]["mode"],
                manifest.agent["reasoning"]["mode"],
            )
            self.assertIsNone(parameters["reasoning"]["effort"])
            self.assertIn(
                parameters["reasoning"]["mode"],
                {"enabled", "disabled", "unsupported"},
            )

    def test_generated_sweep_contains_effective_parameters_and_enforcement(self) -> None:
        manifest = self.load_manifest()

        experiment = manifest.experiment("openrouter", "qa")
        sweep = experiment["model_sweeps"]["openrouter-cypher"]

        self.assertTrue(
            sweep["template"]["phases"]["main"]["client"][
                "require_parameters"
            ]
        )
        self.assertEqual(
            sweep["models"][0]["parameters"], manifest.inference_parameters
        )

    def test_model_parameters_can_omit_unsupported_controls(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        model = manifest.raw["providers"]["ollama"]["models"][0]
        model["parameters"] = {
            "temperature": None,
            "seed": None,
            "reasoning": {"mode": "unsupported", "effort": None},
        }
        model["capabilities"] = {
            "reasoning": "unsupported",
            "reasoning_efforts": None,
            "temperature": False,
            "seed": False,
        }

        manifest.validate()
        self.assertEqual(
            manifest.effective_model_parameters("ollama", model), model["parameters"]
        )

    def test_invalid_reasoning_combination_is_rejected(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["reasoning"] = {
            "mode": "disabled",
            "effort": "high",
        }

        with self.assertRaisesRegex(ManifestError, "only be set"):
            manifest.validate()

    def test_removed_provider_default_effort_has_actionable_error(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["reasoning"]["effort"] = "provider_default"

        with self.assertRaisesRegex(ManifestError, "use null"):
            manifest.validate()

    def test_removed_provider_default_mode_has_actionable_error(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["reasoning"] = {
            "mode": "provider_default",
            "effort": None,
        }

        with self.assertRaisesRegex(ManifestError, "mode: enabled"):
            manifest.validate()

    def test_mandatory_reasoning_cannot_be_disabled(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        model = manifest.raw["providers"]["openrouter"]["models"][0]
        model["parameters"] = {
            "reasoning": {"mode": "disabled", "effort": None}
        }

        with self.assertRaisesRegex(ManifestError, "mandatory reasoning"):
            manifest.validate()

    def test_unsupported_sampling_controls_must_be_null(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        model = manifest.raw["providers"]["ollama"]["models"][0]
        model["capabilities"]["temperature"] = False

        with self.assertRaisesRegex(ManifestError, "temperature unsupported"):
            manifest.validate()

    def test_enabled_reasoning_can_use_default_effort(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["reasoning"]["effort"] = None

        manifest.validate()
        self.assertIsNone(manifest.inference_parameters["reasoning"]["effort"])

    def test_model_default_effort_alias_is_canonicalized_to_null(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["reasoning"]["effort"] = "model_default"

        manifest.validate()
        self.assertIsNone(manifest.inference_parameters["reasoning"]["effort"])

    def test_supported_sampling_controls_must_be_set_explicitly(self) -> None:
        manifest = self.load_manifest()
        manifest.raw = deepcopy(manifest.raw)
        manifest.raw["agent"]["seed"] = None

        with self.assertRaisesRegex(ManifestError, "explicit effective seed"):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()

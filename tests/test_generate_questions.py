import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from scripts import generate_questions as generator
from scripts.question_sampling import Sampler


class QuestionGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = generator.load_catalog(
            generator.ROOT / "data/questions/question_types_v1.yaml"
        )
        cls.scene = generator.Scene.load(generator.DEFAULT_SCENE)

    def build(self, seed=7, catalog=None):
        catalog = catalog or self.catalog
        built = Sampler(self.scene, seed).select(catalog)
        return (
            generator.materialize("qa", catalog["qa"], built["qa"]),
            generator.materialize("pddl", catalog["pddl"], built["pddl"]),
        )

    def test_catalog_wording_can_change_without_changing_grounding(self):
        catalog = deepcopy(self.catalog)
        entry = next(e for e in catalog["qa"] if e["id"] == "qa_object_center_lookup")
        entry["question_template"] = (
            "Coordinates requested for {object_symbol}; return the center."
        )
        original, _ = self.build()
        changed, _ = self.build(catalog=catalog)
        original = next(q for q in original if q["question_type"] == entry["id"])
        changed = next(q for q in changed if q["question_type"] == entry["id"])
        self.assertNotEqual(original["question"], changed["question"])
        self.assertRegex(changed["question"], r"Coordinates requested for O\d+;")
        self.assertEqual(original["solution"], changed["solution"])

    def test_generation_is_reproducible_and_propagates_classification(self):
        qa, pddl = self.build()
        self.assertEqual((qa, pddl), self.build())
        for family, records in (("qa", qa), ("pddl", pddl)):
            for entry, record in zip(self.catalog[family], records):
                self.assertNotIn("name", entry)
                self.assertEqual(record["question_type"], entry["id"])
                self.assertEqual(record["overlap_class"], entry["overlap_class"])
                self.assertTrue(record["name"])
                self.assertNotRegex(record["question"], r"\{[^}]+\}")

    def test_missing_parameter_has_actionable_error(self):
        catalog = deepcopy(self.catalog)
        catalog["qa"][0]["question_template"] = "Return {missing_parameter}."
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.yaml"
            path.write_text(yaml.safe_dump(catalog))
            with self.assertRaisesRegex(
                generator.GenerationError,
                "qa_object_class_inventory.*missing_parameter",
            ):
                generator.load_catalog(path)

    def test_invalid_class_and_template_are_rejected(self):
        for field, value in (
            ("overlap_class", "unknown"),
            ("question_template", "Bad {"),
            ("question_template", "Bad {object.symbol}"),
        ):
            with (
                self.subTest(field=field, value=value),
                TemporaryDirectory() as directory,
            ):
                catalog = deepcopy(self.catalog)
                catalog["qa"][0][field] = value
                path = Path(directory) / "catalog.yaml"
                path.write_text(yaml.safe_dump(catalog))
                with self.assertRaises(generator.GenerationError):
                    generator.load_catalog(path)

    def test_denylist_matches_parameterized_wording_and_catalog_templates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "templates": [
                            {
                                "questions": [
                                    "Where is the center of object {case.object}?"
                                ]
                            }
                        ]
                    }
                )
            )
            for text in (
                " WHERE is the center of object O17? ",
                "Where is the center of object {center_obj}?",
            ):
                with (
                    self.subTest(text=text),
                    self.assertRaisesRegex(
                        generator.GenerationError, "denylisted wording"
                    ),
                ):
                    generator.validate_questions([{"question": text}], [], [path])
            qa, pddl = self.build()
            catalog = deepcopy(self.catalog)
            catalog["qa"][0]["question_template"] = (
                "Where is the center of object {center_obj}?"
            )
            with self.assertRaises(generator.GenerationError):
                generator.validate_questions(qa, pddl, [path], catalog)
            generator.validate_questions(qa, pddl, [path], self.catalog)

    def test_seeded_questions_and_catalog_avoid_available_training_variants(self):
        directory = (
            generator.ROOT.parent / "3dsg-cypher-agent/configs/question_templates"
        )
        paths = [directory / f"cypher_{family}_3dsg.yaml" for family in ("qa", "pddl")]
        if not all(path.is_file() for path in paths):
            self.skipTest("Sibling training catalogs are not available")
        for seed in (1, 7, 42):
            with self.subTest(seed=seed):
                qa, pddl = self.build(seed)
                generator.validate_questions(qa, pddl, paths, self.catalog)


if __name__ == "__main__":
    unittest.main()

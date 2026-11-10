from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

from scripts.generate_questions import Scene, load_catalog
from scripts.question_sampling import Grounder

ROOT = Path(__file__).resolve().parents[1]
HOUSE_SCENE_ID = "habitat_apartment"
QUESTION_DIRECTORY = ROOT / "data" / "questions" / "v1" / HOUSE_SCENE_ID
QUESTION_DIRECTORIES = [
    ROOT / "data/questions" / version / scene_id
    for version in ("v1", "v2")
    for scene_id in (HOUSE_SCENE_ID,)
]


class QuestionArtifactStructureTest(unittest.TestCase):
    def test_saved_bindings_reproduce_questions_answers_and_difficulty_flags(self):
        for directory in QUESTION_DIRECTORIES:
            version = directory.parent.name
            metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
            catalog = load_catalog(
                ROOT / f"data/questions/question_types_{version}.yaml"
            )
            grounder = Grounder(Scene.load(ROOT / metadata["scene_graph"]["path"]))
            for family in ("qa", "pddl"):
                entries = {e["id"]: e for e in catalog[family]}
                records = yaml.safe_load(
                    (directory / f"{family}_questions.yaml").read_text()
                )["questions"]
                for record in records:
                    identifier = record["question_type"]
                    entry = entries[identifier]
                    details = metadata["sampling"]["families"][identifier]
                    with self.subTest(version=version, family=identifier):
                        candidate = grounder.evaluate(entry, details["bindings"])
                        self.assertEqual(
                            candidate.question.solution, record["solution"]
                        )
                        self.assertEqual(
                            entry["question_template"].format_map(
                                candidate.question.parameters
                            ),
                            record["question"],
                        )
                        self.assertEqual(
                            candidate.preferences, details["difficulty_preferences"]
                        )

    def test_versioned_datasets_account_for_every_family_and_match_sources(
        self,
    ) -> None:
        for directory in QUESTION_DIRECTORIES:
            metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
            statuses = metadata["sampling"]["families"]
            self.assertEqual(len(statuses), 100)
            for family in ("qa", "pddl"):
                records = yaml.safe_load(
                    (directory / f"{family}_questions.yaml").read_text()
                )["questions"]
                generated = {
                    key
                    for key, value in statuses.items()
                    if key.startswith(family + "_") and value["status"] == "generated"
                }
                self.assertEqual(generated, {r["question_type"] for r in records})
                self.assertEqual(
                    metadata["coverage"][family]["generated"], len(records)
                )
                self.assertEqual(metadata["coverage"][family]["requested"], 50)
                if directory.name == HOUSE_SCENE_ID:
                    self.assertEqual(len(records), 50)
            for source in ("scene_graph", "question_catalog"):
                info = metadata[source]
                self.assertEqual(
                    info["sha256"],
                    hashlib.sha256((ROOT / info["path"]).read_bytes()).hexdigest(),
                )
            sampler = metadata["generator"]
            self.assertEqual(
                sampler["sampler_sha256"],
                hashlib.sha256(
                    (ROOT / sampler["sampler_path"]).read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                sampler["sha256"],
                hashlib.sha256((ROOT / sampler["path"]).read_bytes()).hexdigest(),
            )

    def test_question_files_contain_no_metadata(self) -> None:
        for filename in ("qa_questions.yaml", "pddl_questions.yaml"):
            with (QUESTION_DIRECTORY / filename).open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)

            self.assertEqual(set(data), {"questions"})
            self.assertEqual(len(data["questions"]), 50)

    def test_metadata_is_kept_in_the_dedicated_file(self) -> None:
        with (QUESTION_DIRECTORY / "metadata.yaml").open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)

        self.assertEqual(metadata["scene_id"], HOUSE_SCENE_ID)
        self.assertEqual(metadata["parameters"]["qa_question_count"], 50)
        self.assertEqual(metadata["parameters"]["pddl_question_count"], 50)
        self.assertEqual(set(metadata["artifacts"]), {"qa", "pddl"})

    def test_smoke_question_set_is_small_and_metadata_free(self) -> None:
        for filename in (
            "qa_questions.smoke.yaml",
            "pddl_questions.smoke.yaml",
        ):
            with (QUESTION_DIRECTORY / filename).open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)

            self.assertEqual(set(data), {"questions"})
            self.assertEqual(len(data["questions"]), 3)

        with (QUESTION_DIRECTORY / "metadata.smoke.yaml").open(
            encoding="utf-8"
        ) as stream:
            metadata = yaml.safe_load(stream)
        self.assertEqual(metadata["scene_id"], HOUSE_SCENE_ID)
        self.assertEqual(metadata["parameters"]["qa_question_count"], 3)
        self.assertEqual(metadata["parameters"]["pddl_question_count"], 3)


if __name__ == "__main__":
    unittest.main()

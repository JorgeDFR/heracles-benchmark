from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUESTION_DIRECTORY = ROOT / "data" / "questions" / "example_dsg"


class QuestionArtifactStructureTest(unittest.TestCase):
    def test_question_files_contain_no_metadata(self) -> None:
        for filename in ("qa_questions.yaml", "pddl_questions.yaml"):
            with (QUESTION_DIRECTORY / filename).open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)

            self.assertEqual(set(data), {"questions"})
            self.assertEqual(len(data["questions"]), 50)

    def test_metadata_is_kept_in_the_dedicated_file(self) -> None:
        with (QUESTION_DIRECTORY / "metadata.yaml").open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)

        self.assertEqual(metadata["scene_id"], "example_dsg")
        self.assertEqual(metadata["parameters"]["qa_question_count"], 50)
        self.assertEqual(metadata["parameters"]["pddl_question_count"], 50)
        self.assertEqual(set(metadata["artifacts"]), {"qa", "pddl"})


if __name__ == "__main__":
    unittest.main()

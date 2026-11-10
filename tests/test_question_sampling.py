"""Semantic regression tests for typed generation, using small independent scenes."""

import re
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

try:
    from lark import Lark
except ImportError:
    Lark = None

from scripts.generate_questions import (
    DEFAULT_SCENE,
    ROOT,
    Entity,
    GenerationError,
    Question,
    Scene,
    load_catalog,
    main,
    materialize,
)
from scripts.question_sampling import (
    Candidate,
    Grounder,
    Ineligible,
    Sampler,
    noun_form,
    pair_answer,
    parameter_role,
)


def entity(symbol, kind, label=None, parents=(), neighbors=(), position=(0, 0, 0)):
    return Entity(
        symbol,
        kind,
        label,
        position,
        frozenset(parents),
        frozenset(neighbors),
        frozenset(),
    )


def fixture():
    return Scene.from_entities(
        [
            entity("R1", "room", "lounge", neighbors=["R2"]),
            entity("R2", "room", "lounge", neighbors=["R3"]),
            entity("R3", "room", "hallway"),
            entity("P1", "mesh_place", parents=["R1"], neighbors=["P2", "p1"]),
            entity("P2", "mesh_place", parents=["R2"], neighbors=["P3"]),
            entity(
                "P3",
                "mesh_place",
                parents=["R3"],
                neighbors=["P4"],
                position=(10, 0, 0),
            ),
            entity("P4", "mesh_place", parents=["R3"], position=(20, 0, 0)),
            entity("p1", "place", parents=["R1"], neighbors=["p2"]),
            entity("p2", "place", parents=["R2"]),
            entity("O1", "object", "chair", parents=["P1"], position=(0, 0, 1)),
            entity("O2", "object", "chair", parents=["P1"], position=(1, 0, 1)),
            entity("O3", "object", "chair", parents=["P2"], position=(2, 0, 1)),
            entity("O4", "object", "chair", parents=["P2"], position=(3, 0, 1)),
            entity("O5", "object", "chair", parents=["P3"], position=(100, 0, 5)),
            entity("O6", "object", "box", parents=["P3"], position=(10, 0, 0)),
            entity("O7", "object", "box", parents=["P4"], position=(20, 0, 0)),
            entity("O8", "object", "box", parents=["P1"], position=(0, 0, 0)),
            entity("O9", "object", "bag", parents=["p1"], position=(0, 0, 0)),
        ]
    )


def eval_goal(text, true_facts):
    """Evaluate a generated Boolean goal directly, without normal-form rewriting."""
    values = re.findall(r"\(|\)|[^\s()]+", text)

    def read(index):
        assert values[index] == "("
        head = values[index + 1]
        index += 2
        children = []
        while values[index] != ")":
            if values[index] == "(":
                child, index = read(index)
                children.append(child)
            else:
                children.append(values[index])
                index += 1
        return [head, *children], index + 1

    tree, end = read(0)
    assert end == len(values)

    def evaluate(node):
        head, *children = node
        if head == "and":
            return all(evaluate(c) for c in children)
        if head == "or":
            return any(evaluate(c) for c in children)
        if head == "not":
            return not evaluate(children[0])
        return (head, *children) in true_facts

    return evaluate(tree)


class QuestionSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogs = {
            v: load_catalog(ROOT / "data/questions" / f"question_types_{v}.yaml")
            for v in ("v1", "v2")
        }
        cls.scene = Scene.load(DEFAULT_SCENE)

    def entry(self, identifier):
        return next(
            e for f in self.catalogs["v2"].values() for e in f if e["id"] == identifier
        )

    def test_pair_answers_are_sets_of_canonical_two_item_lists(self):
        answer = pair_answer(
            [("O1021", "O1014"), ("O27", "O17"), ("O17", "O27"), ("O103", "O98")]
        )
        self.assertEqual(answer, "<[O1014, O1021], [O17, O27], [O98, O103]>")
        self.assertEqual(pair_answer([("R10", "R2")]), "<[R2, R10]>")
        for invalid in (("O1", "O1"), ("O1",), ("O1", "O2", "O3")):
            with self.assertRaises(Ineligible):
                pair_answer([invalid])

    @unittest.skipIf(Lark is None, "external comparator requires lark")
    def test_pair_list_order_matches_the_actual_validator(self):
        source = str(ROOT / "external/heracles_agents/src")
        with patch.object(sys, "path", [source, *sys.path]):
            from sldp.sldp_lang import sldp_equals
        self.assertTrue(sldp_equals("<[O2, O10], [O3, O4]>", "<[O3, O4], [O2, O10]>"))
        self.assertFalse(sldp_equals("<[O2, O10]>", "<[O10, O2]>"))
        self.assertTrue(sldp_equals("<[O2, O10]>", "<[O10, O2]>", unordered_pairs=True))
        self.assertTrue(
            sldp_equals("{R1: <[O2, O10]>}", "{R1: <[O10, O2]>}", unordered_pairs=True)
        )
        self.assertFalse(sldp_equals("[R1, R2]", "[R2, R1]"))
        self.assertFalse(sldp_equals("[1, 2]", "[2, 1]", unordered_pairs=True))
        self.assertFalse(sldp_equals("<[O2, O10]>", "<<O2, O10>>"))
        self.assertTrue(sldp_equals("{R1: <[O2, O10]>}", "{R1: <[O2, O10]>}"))

    @unittest.skipIf(Lark is None, "external parser requires lark")
    def test_all_pair_families_emit_canonical_lists(self):
        expected = {
            "v1": {"qa_closest_cross_class_pair", "qa_closest_room_pair"},
            "v2": {
                "qa_closest_class_pairs_in_room_class",
                "qa_closest_pair_in_different_rooms",
                "qa_mutual_nearest_class_pairs",
                "qa_closest_class_pairs_per_room",
            },
        }
        parser = Lark(
            (ROOT / "external/heracles_agents/src/sldp/resources/sldp.lark").read_text()
        )
        for version, identifiers in expected.items():
            sampler = Sampler(self.scene, pool_size=4)
            for entry in self.catalogs[version]["qa"]:
                if entry["id"] not in identifiers:
                    continue
                pool, report = sampler.candidates(entry)
                self.assertEqual(report["status"], "generated", entry["id"])
                for candidate in pool:
                    record = materialize(
                        "qa", [entry], {entry["id"]: candidate.question}
                    )[0]
                    self.assertTrue(record["correctness_comparator"]["unordered_pairs"])
                    tree = parser.parse(candidate.question.solution)
                    lists = list(tree.find_data("list"))
                    self.assertTrue(lists, entry["id"])
                    for pair in lists:
                        self.assertEqual(len(pair.children), 2)
                        values = [str(node.children[0]) for node in pair.children]
                        self.assertLess(int(values[0][1:]), int(values[1][1:]))
                    for group in tree.find_data("set"):
                        self.assertTrue(
                            all(child.data == "list" for child in group.children)
                        )

    def test_external_reference_selection_uses_nearest_in_each_scope(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room", "bedroom"),
                entity("R2", "room", "lounge"),
                entity("R3", "room", "study"),
                entity("P1", "mesh_place", parents=["R1"]),
                entity("P2", "mesh_place", parents=["R2"]),
                entity("P3", "mesh_place", parents=["R3"]),
                entity("O1", "object", "chair", parents=["P1"], position=(0, 0, 0)),
                entity("O2", "object", "chair", parents=["P2"], position=(2, 0, 0)),
                entity("O3", "object", "window", parents=["P1"], position=(10, 0, 0)),
                entity("O4", "object", "window", parents=["P2"], position=(1, 0, 0)),
                entity("O5", "object", "chair", parents=["P3"]),
                entity("O6", "object", "window", position=(2, 0, 0)),
            ]
        )
        binding = {"target_object_class": "chair", "reference_object_class": "window"}
        result = Grounder(scene).evaluate(
            self.entry("qa_targets_nearer_external_reference"), binding
        )
        self.assertEqual(result.question.solution, "<O1>")
        # A nearer local reference excludes O1 even though another local one is far.
        scene = Scene.from_entities(
            list(scene.entities.values())
            + [entity("O7", "object", "window", parents=["P1"], position=(0.1, 0, 0))]
        )
        with self.assertRaises(Ineligible):
            Grounder(scene).evaluate(
                self.entry("qa_targets_nearer_external_reference"), binding
            )

    def test_external_reference_distance_ties_are_not_closer(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room"),
                entity("R2", "room"),
                entity("P1", "mesh_place", parents=["R1"]),
                entity("P2", "mesh_place", parents=["R2"]),
                entity("O1", "object", "chair", parents=["P1"]),
                entity("O2", "object", "chair", parents=["P2"], position=(10, 0, 0)),
                entity("O3", "object", "window", parents=["P1"], position=(1, 0, 0)),
                entity("O4", "object", "window", parents=["P2"], position=(-1, 0, 0)),
            ]
        )
        result = Grounder(scene).evaluate(
            self.entry("qa_targets_nearer_external_reference"),
            {"target_object_class": "chair", "reference_object_class": "window"},
        )
        self.assertEqual(result.question.solution, "<O2>")

    def test_universal_room_visits_apply_content_exclusion_not_visit_prohibition(self):
        nodes = [
            entity(f"R{i}", "room", "bedroom" if i < 4 else "study")
            for i in range(1, 5)
        ]
        nodes += [entity(f"P{i}", "mesh_place", parents=[f"R{i}"]) for i in range(1, 5)]
        nodes += [
            entity(f"O{i}", "object", "lamp", parents=[f"P{i}"]) for i in range(1, 5)
        ]
        nodes += [entity("O5", "object", "desk", parents=["P3"])]
        result = Grounder(Scene.from_entities(nodes)).evaluate(
            self.entry("pddl_visit_target_rooms_without_excluded"),
            {
                "source_room_class": "bedroom",
                "target_object_class": "lamp",
                "excluded_object_class": "desk",
            },
        )
        goal = result.question.solution
        self.assertTrue(
            eval_goal(goal, {("visited-region", "R1"), ("visited-region", "R2")})
        )
        self.assertFalse(
            eval_goal(
                goal,
                {
                    ("visited-region", "R1"),
                    ("visited-region", "R3"),
                    ("visited-region", "R4"),
                },
            )
        )
        self.assertTrue(
            eval_goal(goal, {("visited-region", f"R{i}") for i in range(1, 5)})
        )

    def test_light_uses_simple_natural_noun_forms(self):
        for typ, count, expected in (
            ("object_class_singular", None, "light"),
            ("object_class_plural", None, "lights"),
            ("object_class_indefinite", None, "a light"),
            ("object_class_counted", 1, "light"),
            ("object_class_counted", 2, "lights"),
        ):
            self.assertEqual(noun_form(fixture(), "light", typ, count), expected)

    def test_catalogs_have_same_schema_and_no_versioned_ids(self):
        expected = {
            "id",
            "description",
            "question_template",
            "overlap_class",
            "tags",
            "parameters",
            "requires",
        }
        for catalog in self.catalogs.values():
            for entries in catalog.values():
                self.assertEqual(len(entries), 50)
                for entry in entries:
                    self.assertEqual(set(entry), expected)
                    self.assertNotIn("_v2_", entry["id"])
        for entries in self.catalogs["v2"].values():
            for entry in entries:
                self.assertTrue(
                    callable(getattr(Grounder, entry["id"], None)), entry["id"]
                )
                self.assertNotRegex(
                    entry["question_template"],
                    r"\{[^}]*_class_[^}]+\}\s+(objects?|rooms?)\b",
                )

    def test_reproducible_joint_sampling_and_coverage(self):
        sampler = Sampler(self.scene, 42, pool_size=12)
        built = sampler.select(self.catalogs["v2"])
        repeat = Sampler(self.scene, 42, pool_size=12)
        self.assertEqual(built, repeat.select(self.catalogs["v2"]))
        self.assertEqual(sampler.report, repeat.report)
        self.assertEqual(len(sampler.report["families"]), 100)
        self.assertTrue(
            any(
                d["status"] == "unsupported"
                for d in sampler.report["families"].values()
            )
        )
        self.assertGreater(len(built["qa"]), 25)
        self.assertGreater(len(built["pddl"]), 25)
        self.assertGreater(
            len(
                sampler.report["diversity"]["class_role_counts"]["target_object_class"]
            ),
            5,
        )
        for family in ("qa", "pddl"):
            records = materialize(
                family, self.catalogs["v2"][family], built[family], allow_partial=True
            )
            self.assertEqual(len(records), len(built[family]))
            self.assertTrue(all(record["solution"] for record in records))
        another = Sampler(self.scene, 43, pool_size=12).select(self.catalogs["v2"])
        self.assertNotEqual(built, another)

    def test_class_forms_share_one_binding_and_support_custom_ontology(self):
        self.assertEqual(
            parameter_role("source_room_class_singular", "room_class_singular"),
            parameter_role("source_room_class_plural", "room_class_plural"),
        )
        scene = fixture()
        scene._metadata = {
            "question_generation": {
                "noun_forms": {
                    "equipment": {
                        "singular": "piece of equipment",
                        "plural": "pieces of equipment",
                    }
                }
            }
        }
        self.assertEqual(
            noun_form(scene, "equipment", "object_class_plural"), "pieces of equipment"
        )
        self.assertEqual(
            noun_form(scene, "equipment", "object_class_indefinite"),
            "a piece of equipment",
        )

    def test_containment_and_adjacency_include_layers_without_crossing_them(self):
        scene = fixture()
        g = Grounder(scene)
        g.binding = {"source_room_class": "lounge", "reference_object_class": "box"}
        self.assertEqual(g.qa_place_counts_in_content_selected_room(), "2")
        self.assertIn("p1", scene.places_by_room["R1"])
        self.assertIn("P1", scene.places_by_room["R1"])
        self.assertEqual(g.pa["P1"], {"P2"})
        self.assertIn("P1", g.pa["P2"])
        self.assertNotIn("p1", g.pa["P1"])
        self.assertEqual(set(scene.rooms_for_object("O9")), {"R1"})
        nested = Scene.from_entities(
            [
                entity("R1", "room"),
                entity("p1", "place", parents=["R1"]),
                entity("P1", "mesh_place", parents=["p1"]),
                entity("O1", "object", parents=["P1"]),
            ]
        )
        self.assertEqual(nested.rooms_for_object("O1"), ["R1"])

    def test_global_farthest_search_is_not_restricted_to_anchor_room(self):
        g = Grounder(fixture())
        g.binding = {
            "target_object_class": "chair",
            "reference_object_class": "bag",
            "source_room_class": "lounge",
        }
        # Add a same-class reference outside the scoped category, so the room
        # reference requires containment disambiguation.
        scene = Scene.from_entities(
            list(g.scene.entities.values())
            + [entity("O10", "object", "bag", parents=["P3"])]
        )
        g = Grounder(scene)
        g.binding = {
            "target_object_class": "chair",
            "reference_object_class": "bag",
            "source_room_class": "lounge",
        }
        self.assertEqual(g.qa_global_farthest_from_selected_room(), "<O5>")
        self.assertEqual(g.oroom["O9"], {"R1"})

    def test_reference_resolution_does_not_mutate_multiple_room_memberships(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room", "lounge"),
                entity("R2", "room", "hallway"),
                entity("P1", "mesh_place", parents=["R1", "R2"]),
                entity("O1", "object", "bag", parents=["P1"]),
                entity("P2", "mesh_place", parents=["R2"]),
                entity("O2", "object", "bag", parents=["P2"]),
            ]
        )
        g = Grounder(scene)
        g.binding = {"reference_object_class": "bag", "source_room_class": "lounge"}
        self.assertEqual(g.anchor_room(room_role="source"), "R1")
        self.assertEqual(g.oroom["O1"], {"R1", "R2"})

    def test_pddl_preserves_shared_and_independent_destination_scope(self):
        g = Grounder(fixture())
        binding = {
            "target_object_class": "chair",
            "reference_object_class": "box",
            "source_room_class": "lounge",
            "destination_room_class": "hallway",
        }
        shared = g.evaluate(
            self.entry("pddl_move_all_to_one_shared_destination"), binding
        ).question.solution
        independent = g.evaluate(
            self.entry("pddl_move_each_to_independent_destination"), binding
        ).question.solution
        mixed = {
            ("object-in-place", o, "P3" if o in {"O1", "O2"} else "P4")
            for o in ["O1", "O2", "O3", "O4"]
        }
        self.assertFalse(eval_goal(shared, mixed))
        self.assertTrue(eval_goal(independent, mixed))
        together = {("object-in-place", o, "P3") for o in ["O1", "O2", "O3", "O4"]}
        self.assertTrue(eval_goal(shared, together))
        self.assertTrue(eval_goal(independent, together))

    def test_pddl_inspection_and_holding_require_same_identity(self):
        g = Grounder(fixture())
        clause = g.evaluate(
            self.entry("pddl_inspect_and_hold_same_choice"),
            {"target_object_class": "chair", "source_room_class": "lounge"},
        ).question.solution
        self.assertFalse(eval_goal(clause, {("safe", "O1"), ("holding", "O2")}))
        self.assertTrue(eval_goal(clause, {("safe", "O1"), ("holding", "O1")}))

    def test_hop_zero_is_included_and_reference_layer_is_unambiguous(self):
        scene = Scene.from_entities(
            list(fixture().entities.values())
            + [
                entity("O10", "object", "chair", parents=["P4"]),
                entity("O11", "object", "bag", parents=["P3"]),
            ]
        )
        g = Grounder(scene)
        g.binding = {
            "target_object_class": "chair",
            "reference_object_class": "box",
            "source_room_class": "lounge",
            "place_layer_name": "mesh_place",
            "hop_limit": 2,
        }
        self.assertEqual(g.hop_targets(), {"O1", "O2", "O3", "O4", "O5"})
        g.binding["reference_object_class"] = "bag"
        with self.assertRaisesRegex(Ineligible, "direct parent"):
            g.hop_targets()

    def test_budget_exhaustion_is_not_claimed_as_scene_unsupported(self):
        sampler = Sampler(self.scene, max_attempts=1)
        entry = self.entry("qa_shortest_room_route_avoiding_class")
        pool, status = sampler.candidates(entry)
        self.assertFalse(pool)
        self.assertEqual(status["status"], "search_exhausted")
        self.assertFalse(status["search_complete"])

    def test_grouped_and_exact_counts_accept_zero_one_distributions(self):
        scene = Scene.from_entities(
            [entity(f"R{i}", "room", "bedroom") for i in range(1, 4)]
            + [entity(f"P{i}", "mesh_place", parents=[f"R{i}"]) for i in range(1, 4)]
            + [
                entity("O1", "object", "desk", parents=["P2"]),
                entity("O2", "object", "desk", parents=["P3"]),
            ]
        )
        g = Grounder(scene)
        binding = {"source_room_class": "bedroom", "target_object_class": "desk"}
        result = g.evaluate(self.entry("qa_counts_per_room"), binding)
        self.assertEqual(result.question.solution, "{R1: 0, R2: 1, R3: 1}")
        self.assertFalse(result.preferences["unequal_positive_counts"])
        result = g.evaluate(
            self.entry("pddl_visit_rooms_with_exact_target_count"),
            {**binding, "required_count": 1},
        )
        self.assertTrue(
            eval_goal(
                result.question.solution,
                {("visited-region", "R2"), ("visited-region", "R3")},
            )
        )
        with self.assertRaises(Ineligible):
            g.evaluate(
                self.entry("qa_rooms_with_exact_count"),
                {**binding, "required_count": 2},
            )
        sampler = Sampler(scene)
        values = list(
            sampler.numeric_bindings(
                self.entry("qa_rooms_with_exact_count"), binding, sampler.rng("count")
            )
        )
        self.assertEqual([v["required_count"] for v in values], [1])
        self.assertEqual(
            result.question.parameters["target_object_class_counted"], "desk"
        )

    def test_class_intersection_needs_no_matching_room_outside_category(self):
        scene = Scene.from_entities(
            [entity(f"R{i}", "room", "bedroom") for i in range(1, 4)]
            + [entity(f"P{i}", "mesh_place", parents=[f"R{i}"]) for i in range(1, 4)]
            + [
                entity("O1", "object", "desk", parents=["P1"]),
                entity("O2", "object", "desk", parents=["P2"]),
                entity("O3", "object", "book", parents=["P2"]),
                entity("O4", "object", "book", parents=["P3"]),
            ]
        )
        result = Grounder(scene).evaluate(
            self.entry("qa_rooms_with_both_classes"),
            {
                "source_room_class": "bedroom",
                "target_object_class": "desk",
                "reference_object_class": "book",
            },
        )
        self.assertEqual(result.question.solution, "<R2>")
        self.assertFalse(result.preferences["room_category_excludes_matching_room"])

    def test_per_room_answers_keep_singleton_rows_and_boolean_scope(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room", "bedroom"),
                entity("R2", "room", "bedroom"),
                entity("P1", "mesh_place", parents=["R1"]),
                entity("P2", "mesh_place", parents=["R2"]),
                entity("O1", "object", "chair", parents=["P1"], position=(0, 0, 0)),
                entity("O2", "object", "chair", parents=["P1"], position=(2, 0, 0)),
                entity("O3", "object", "chair", parents=["P2"], position=(6, 0, 0)),
                entity("O4", "object", "lamp", parents=["P1"], position=(0, 0, 1)),
                entity("O5", "object", "lamp", parents=["P2"], position=(6, 0, 1)),
            ]
        )
        g = Grounder(scene)
        b = {
            "source_room_class": "bedroom",
            "target_object_class": "chair",
            "reference_object_class": "lamp",
        }
        self.assertEqual(
            g.evaluate(self.entry("qa_nearest_targets_per_room"), b).question.solution,
            "{R1: <O1>, R2: <O3>}",
        )
        visit = g.evaluate(
            self.entry("pddl_visit_one_target_per_room"), b
        ).question.solution
        self.assertTrue(
            eval_goal(visit, {("visited-object", "O2"), ("visited-object", "O3")})
        )
        self.assertFalse(
            eval_goal(visit, {("visited-object", "O1"), ("visited-object", "O2")})
        )
        complete = g.evaluate(
            self.entry("pddl_choose_room_choose_complete_task"), b
        ).question.solution
        self.assertFalse(eval_goal(complete, {("safe", "O1")}))
        self.assertTrue(eval_goal(complete, {("safe", "O1"), ("safe", "O2")}))
        self.assertTrue(eval_goal(complete, {("safe", "O3")}))
        self.assertTrue(eval_goal(complete, {("visited-object", "O4")}))

    def test_adaptive_hops_and_reference_domains_on_dense_navigation(self):
        nodes = [entity("R1", "room", "kitchen")]
        nodes += [
            entity(
                f"P{i}",
                "mesh_place",
                parents=["R1"],
                neighbors=[f"P{i + 1}"] if i < 25 else [],
                position=(i * 0.2, 0, 0),
            )
            for i in range(1, 26)
        ]
        nodes += [entity("O1", "object", "ceiling_light", parents=["P1"])]
        nodes += [
            entity(f"O{i}", "object", "chair", parents=[f"P{p}"])
            for i, p in enumerate((3, 6, 12, 25), 2)
        ]
        scene = Scene.from_entities(nodes)
        sampler = Sampler(scene)
        entry = self.entry("qa_target_count_within_place_hops")
        bindings = list(sampler.base_bindings(entry, sampler.rng(entry["id"])))
        self.assertTrue(bindings)
        self.assertTrue(
            all(b["reference_object_class"] == "ceiling_light" for b in bindings)
        )
        numeric = list(
            sampler.numeric_bindings(entry, bindings[0], sampler.rng("numeric"))
        )
        self.assertIn(11, [b["hop_limit"] for b in numeric])
        candidates, report = sampler.candidates(entry)
        self.assertEqual(report["status"], "generated")
        self.assertEqual(candidates[0].question.solution, "3")
        self.assertGreater(candidates[0].bindings["hop_limit"], 6)

    def test_radius_domain_includes_small_scoped_thresholds(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room", "bedroom"),
                entity("R2", "room", "bedroom"),
                entity("P1", "mesh_place", parents=["R1"]),
                entity("P2", "mesh_place", parents=["R2"]),
                entity("O1", "object", "lamp", parents=["P1"], position=(0.3, 0, 0)),
                entity("O2", "object", "lamp", parents=["P1"], position=(0.5, 0, 0)),
                entity("O3", "object", "lamp", parents=["P2"], position=(10.3, 0, 0)),
                entity("O4", "object", "bed", parents=["P1"]),
                entity("O5", "object", "bed", parents=["P2"], position=(10, 0, 0)),
            ]
        )
        sampler = Sampler(scene)
        b = {
            "source_room_class": "bedroom",
            "target_object_class": "lamp",
            "reference_object_class": "bed",
        }
        values = list(
            sampler.numeric_bindings(
                self.entry("qa_rooms_with_all_targets_near_references"),
                b,
                sampler.rng("radius"),
            )
        )
        self.assertIn(0.4, [v["radius_m"] for v in values])
        candidate = sampler.grounder.evaluate(
            self.entry("qa_rooms_with_all_targets_near_references"),
            {**b, "radius_m": 0.4},
        )
        self.assertEqual(candidate.question.solution, "<R2>")

    def test_legacy_symbol_anchored_hops_still_generate(self):
        entry = next(
            e for e in self.catalogs["v1"]["qa"] if e["id"] == "qa_places_within_hops"
        )
        pool, report = Sampler(fixture()).candidates(entry)
        self.assertEqual(report["status"], "generated")
        self.assertTrue(pool)
        self.assertTrue(all(2 <= c.bindings["hop_limit"] <= 6 for c in pool))

    def test_numerical_selection_prefers_two_sided_count_contrast(self):
        nodes = [entity(f"R{i}", "room", "bedroom") for i in range(1, 4)]
        nodes += [entity(f"P{i}", "mesh_place", parents=[f"R{i}"]) for i in range(1, 4)]
        index = 1
        for room, count in ((1, 1), (2, 2), (3, 3)):
            for _ in range(count):
                nodes.append(
                    entity(f"O{index}", "object", "book", parents=[f"P{room}"])
                )
                index += 1
        pool, report = Sampler(Scene.from_entities(nodes)).candidates(
            self.entry("qa_rooms_with_exact_count")
        )
        self.assertEqual(report["status"], "generated")
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].bindings["required_count"], 2)
        self.assertEqual(pool[0].question.solution, "<R2>")
        self.assertEqual(
            pool[0].question.parameters["target_object_class_counted"], "books"
        )

    def test_invalid_selected_layer_is_a_preparation_rejection(self):
        sampler = Sampler(fixture())
        pool, report = sampler.candidates(self.entry("qa_targets_within_place_hops"))
        self.assertIn(
            "reference has no unique direct parent in the selected layer",
            report["preparation_rejections"],
        )
        self.assertTrue(
            all(
                candidate.bindings["place_layer_name"] == "mesh_place"
                for candidate in pool
            )
        )

    def test_unchanged_routes_and_mandatory_intermediates_remain_valid(self):
        scene = Scene.from_entities(
            [
                entity("R1", "room", "normal", neighbors=["R2"]),
                entity("R2", "room", "normal", neighbors=["R3"]),
                entity("R3", "room", "normal"),
                entity("R4", "room", "blocked"),
                entity("P1", "mesh_place", parents=["R1"], neighbors=["P2"]),
                entity(
                    "P2",
                    "mesh_place",
                    parents=["R2"],
                    neighbors=["P3"],
                    position=(0.2, 0, 0),
                ),
                entity("P3", "mesh_place", parents=["R3"], position=(0.4, 0, 0)),
                entity("O1", "object", "bicycle", parents=["P1"]),
                entity("O2", "object", "bookcase", parents=["P3"]),
            ]
        )
        g = Grounder(scene)
        b = {
            "reference_object_class": "bicycle",
            "destination_object_class": "bookcase",
            "excluded_room_class": "blocked",
            "place_layer_name": "mesh_place",
        }
        self.assertEqual(
            g.evaluate(self.entry("qa_room_avoidance_detour"), b).question.solution, "0"
        )
        self.assertEqual(
            g.evaluate(
                self.entry("qa_shortest_room_route_avoiding_class"), b
            ).question.solution,
            "[R1, R2, R3]",
        )
        self.assertEqual(
            g.evaluate(
                self.entry("qa_critical_rooms_between_anchors"), b
            ).question.solution,
            "<R2>",
        )
        weighted = g.evaluate(self.entry("qa_minimum_length_place_route"), b)
        self.assertEqual(weighted.question.solution, "0.40")
        self.assertFalse(weighted.preferences["weighted_differs_from_hop_optimum"])

    def test_selection_prefers_stronger_candidates_and_records_fallbacks(self):
        weak = Candidate(
            Question({}, "weak"), {}, frozenset(), frozenset(), {"contrast": False}
        )
        strong = Candidate(
            Question({}, "strong"), {}, frozenset(), frozenset(), {"contrast": True}
        )

        class FixedSampler(Sampler):
            def candidates(self, entry):
                return [weak, strong], {"status": "generated"}

        sampler = FixedSampler(fixture())
        chosen = sampler.select({"qa": [{"id": "qa_test"}], "pddl": []})
        self.assertEqual(chosen["qa"]["qa_test"].solution, "strong")
        self.assertEqual(
            sampler.report["families"]["qa_test"]["difficulty_preferences"],
            {"contrast": True},
        )

    def test_strict_mode_does_not_write_partial_artifacts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "spark_dsg_example"
            code = main(
                [
                    "--catalog",
                    str(ROOT / "data/questions/question_types_v2.yaml"),
                    "--output-dir",
                    str(output),
                    "--strict",
                ]
            )
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())

    def test_bad_parameter_types_and_capabilities_are_rejected(self):
        for field, value in [
            (
                "parameters",
                {
                    "target_object_class_plural": "not_a_type",
                    "source_room_class_plural": "room_class_plural",
                },
            ),
            ("requires", ["made_up"]),
        ]:
            catalog = deepcopy(self.catalogs["v2"])
            catalog["qa"][0][field] = value
            with TemporaryDirectory() as directory:
                p = Path(directory) / "bad.yaml"
                p.write_text(yaml.safe_dump(catalog))
                with self.assertRaises(GenerationError):
                    load_catalog(p)

    def test_constrained_routes_components_and_critical_rooms(self):
        links = {
            "R9": ["R1"],
            "R1": ["R2", "R5"],
            "R2": ["R3", "R8", "R10"],
            "R3": ["R4"],
            "R5": ["R6"],
            "R6": ["R7"],
            "R7": ["R4"],
            "R4": [],
            "R8": [],
            "R10": [],
        }
        nodes = [
            entity(r, "room", "blocked" if r == "R2" else "normal", neighbors=n)
            for r, n in links.items()
        ]
        for index, (room, label) in enumerate(
            [
                ("R9", "bag"),
                ("R4", "bed"),
                ("R3", "chair"),
                ("R8", "chair"),
                ("R4", "chair"),
            ],
            1,
        ):
            nodes += [
                entity(f"P{index}", "mesh_place", parents=[room]),
                entity(f"O{index}", "object", label, parents=[f"P{index}"]),
            ]
        g = Grounder(Scene.from_entities(nodes))
        g.binding = {
            "reference_object_class": "bag",
            "destination_object_class": "bed",
            "excluded_room_class": "blocked",
            "target_object_class": "chair",
            "source_room_class": "blocked",
            "comparison_room_class": "normal",
        }
        self.assertEqual(g.qa_room_hops_between_content_anchors(), "4")
        self.assertEqual(g.qa_room_avoidance_detour(), "1")
        self.assertEqual(
            g.qa_shortest_room_route_avoiding_class(), "[R9, R1, R5, R6, R7, R4]"
        )
        self.assertEqual(g.qa_critical_rooms_between_anchors(), "<R1>")
        self.assertEqual(g.qa_room_components_after_category_removal(), "3")
        self.assertEqual(g.qa_reachable_target_rooms_avoiding_class(), "<R3, R4>")
        clause = g.pddl_inspect_targets_in_allowed_room_component()
        self.assertTrue(eval_goal(clause, {("safe", "O3"), ("safe", "O5")}))
        self.assertFalse(eval_goal(clause, {("safe", "O3"), ("safe", "O4")}))
        self.assertTrue(g.qa_rooms_adjacent_to_two_room_classes())

    def test_weighted_shortest_route_is_not_minimum_hop_route(self):
        nodes = [
            entity("R1", "room"),
            entity(
                "P1",
                "mesh_place",
                parents=["R1"],
                neighbors=["P2", "P4"],
                position=(0, 0, 0),
            ),
            entity(
                "P2",
                "mesh_place",
                parents=["R1"],
                neighbors=["P3"],
                position=(0, 10, 0),
            ),
            entity("P3", "mesh_place", parents=["R1"], position=(3, 0, 0)),
            entity(
                "P4", "mesh_place", parents=["R1"], neighbors=["P5"], position=(1, 0, 0)
            ),
            entity(
                "P5", "mesh_place", parents=["R1"], neighbors=["P3"], position=(2, 0, 0)
            ),
            entity("O1", "object", "bag", parents=["P1"]),
            entity("O2", "object", "bed", parents=["P3"]),
        ]
        g = Grounder(Scene.from_entities(nodes))
        g.binding = {
            "reference_object_class": "bag",
            "destination_object_class": "bed",
            "place_layer_name": "mesh_place",
        }
        self.assertEqual(g.qa_minimum_length_place_route(), "3.00")

    def test_universal_class_inventory_intersects_instead_of_unions(self):
        nodes = []
        for index, extra in enumerate(["box", "bag", "light"], 1):
            nodes += [
                entity(f"R{index}", "room", "hallway"),
                entity(f"P{index}", "mesh_place", parents=[f"R{index}"]),
                entity(f"O{2 * index}", "object", "chair", parents=[f"P{index}"]),
                entity(f"O{2 * index + 1}", "object", extra, parents=[f"P{index}"]),
            ]
        g = Grounder(Scene.from_entities(nodes))
        g.binding = {"source_room_class": "hallway"}
        self.assertEqual(g.qa_classes_in_every_room(), "<chair>")

    @unittest.skipIf(Lark is None, "external parser grammar validation requires lark")
    def test_both_datasets_parse_with_external_validator_grammars(self):
        source = ROOT / "external/heracles_agents/src"
        parsers = {
            "qa": Lark((source / "sldp/resources/sldp.lark").read_text()),
            "pddl": Lark(
                (source / "pypddl/resources/pddl_goal.lark").read_text(), parser="lalr"
            ),
        }
        for version in ("v1", "v2"):
            built = Sampler(self.scene, pool_size=12).select(self.catalogs[version])
            for family, questions in built.items():
                for identifier, question in questions.items():
                    with self.subTest(version=version, identifier=identifier):
                        parsers[family].parse(question.solution)


if __name__ == "__main__":
    unittest.main()

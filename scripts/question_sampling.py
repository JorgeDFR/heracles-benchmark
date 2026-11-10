"""Typed candidate grounding and deterministic diversity selection for both catalogs."""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import re
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, pairwise, product
from typing import Any

from scripts.generate_questions import (
    GenerationError,
    Question,
    Scene,
    format_number,
    format_point,
    indefinite,
    pddl_fact,
    pddl_group,
    sldp_list,
    sldp_set,
    sorted_symbols,
)

PARAMETER_TYPES = frozenset(
    {
        "object_class_singular",
        "object_class_plural",
        "object_class_indefinite",
        "object_class_counted",
        "room_class_singular",
        "room_class_plural",
        "room_class_indefinite",
        "object_symbol",
        "room_symbol",
        "place_symbol",
        "distance_m",
        "positive_integer",
        "place_layer_name",
    }
)
CAPABILITIES = frozenset(
    {
        "object_classes",
        "room_classes",
        "containment",
        "navigable_places",
        "object_centers",
        "room_centers",
        "place_centers",
        "vertical_axis",
        "room_connectivity",
        "place_connectivity",
        "selected_place_layer",
        "selected_place_connectivity",
    }
)
# These phrases retain the labelspace's category breadth; no random aliases.
NOUN_FORMS = {
    "seating": ("seat", "seats"),
    "storage": ("storage unit", "storage units"),
    "decor": ("decoration", "decorations"),
    "trash": ("trash can", "trash cans"),
    "food": ("food item", "food items"),
    "light": ("light", "lights"),
}
TIE_TOLERANCE = 1e-6

# Reference semantics belong to the handlers, independently of template wording.
# None means globally unique; a room role means unique across that room category.
REFERENCE_SCOPES = {
    **{
        key: {"reference": "source"}
        for key in (
            "qa_place_counts_in_content_selected_room",
            "qa_nearest_to_room_disambiguated_reference",
            "qa_global_farthest_from_selected_room",
            "qa_targets_within_place_hops",
            "qa_target_count_within_place_hops",
            "pddl_inspect_hop_reachable_targets",
            "pddl_end_at_minimum_hop_reference_place",
            "pddl_hold_minimum_hop_target",
        )
    },
    **{
        key: {"reference": None, "destination": None}
        for key in (
            "qa_room_hops_between_content_anchors",
            "qa_shortest_room_route_avoiding_class",
            "qa_critical_rooms_between_anchors",
            "qa_minimum_length_place_route",
            "qa_room_avoidance_detour",
        )
    },
    **{
        key: {"reference": None}
        for key in (
            "qa_reachable_target_rooms_avoiding_class",
            "qa_targets_in_distance_band",
            "qa_nearest_targets_outside_reference_room",
            "qa_target_rooms_within_room_hops",
            "qa_highest_target_in_selected_room",
            "pddl_inspect_targets_above_reference",
            "pddl_end_adjacent_to_reference_avoid_category",
            "pddl_visit_nearby_locations_except_start",
            "pddl_visit_one_per_room_end_at_reference",
            "pddl_inspect_targets_in_allowed_room_component",
        )
    },
    **{
        key: {"reference": None, "comparison": None}
        for key in (
            "qa_targets_closer_to_first_reference",
            "qa_common_place_hop_neighborhood",
            "pddl_hold_target_closer_to_first_reference",
        )
    },
    "pddl_move_unique_target_to_reference_place": {
        "target": "source",
        "reference": "destination",
    },
    "pddl_move_one_target_per_source_room": {"reference": "destination"},
    "pddl_keep_target_out_of_reference_places": {"target": None},
    "pddl_move_target_to_minimum_hop_reference": {"target": None},
}


class Ineligible(Exception):
    """An assignment does not meet a family's semantic/contrast requirements."""


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise Ineligible(reason)


def plural(noun: str) -> str:
    """Conservative count-noun inflection; metadata overrides handle other ontologies."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def noun_form(scene: Scene, name: str, typ: str, count: int | None = None) -> str:
    forms = scene._metadata.get("question_generation", {}).get("noun_forms", {})
    custom = forms.get(name)
    if custom is not None:
        require(
            isinstance(custom, dict) and {"singular", "plural"} <= custom.keys(),
            f"incomplete noun forms for {name}",
        )
        singular, multiple = custom["singular"], custom["plural"]
        require(
            all(isinstance(v, str) and v.strip() for v in (singular, multiple)),
            f"invalid noun forms for {name}",
        )
    else:
        # Leave underscores out of spoken names, but retain canonical class bindings.
        readable = name.replace("_", " ").lower()
        singular, multiple = NOUN_FORMS.get(readable, (readable, plural(readable)))
    if typ.endswith("_counted"):
        require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "counted noun requires a positive required_count",
        )
        return singular if count == 1 else multiple
    if typ.endswith("_plural"):
        return multiple
    if typ.endswith("_indefinite"):
        return indefinite(singular)
    return singular


def parameter_role(name: str, typ: str) -> str:
    if "_class_" in typ:
        return re.sub(r"_(singular|plural|indefinite|counted)$", "", name)
    return name


def set_answer(values: Iterable[str]) -> str:
    return sldp_set(sorted_symbols(values))


def class_answer(values: Iterable[str]) -> str:
    values = sorted(set(values))
    require(
        all(re.fullmatch(r"[A-Za-z_]\w*", value) for value in values),
        "class labels are not valid SLDP tokens",
    )
    return sldp_set(values)


def pair_answer(pairs: Iterable[tuple[str, str]]) -> str:
    # Keep reference serialization deterministic. Pair-aware validation accepts
    # either endpoint order; this display convention is not a model requirement.
    formatted = set()
    for pair in pairs:
        endpoints = sorted_symbols(pair)
        require(len(endpoints) == 2, "a pair requires two distinct endpoints")
        formatted.add(sldp_list(endpoints))
    return sldp_set(sorted(formatted))


def dictionary(values: dict[str, str | int]) -> str:
    require(
        all(re.fullmatch(r"[A-Za-z_]\w*", key) for key in values),
        "dictionary keys are not valid SLDP tokens",
    )
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(values.items())) + "}"


def fold(operator: str, clauses: Iterable[str]) -> str:
    return pddl_group(operator, list(dict.fromkeys(clauses)))


def facts(predicate: str, values: Iterable[str], operator: str = "or") -> str:
    return fold(operator, (pddl_fact(predicate, x) for x in sorted_symbols(values)))


def negated(predicate: str, *args: str) -> str:
    return f"(not ({predicate} {' '.join(args)}))"


@dataclass
class Candidate:
    question: Question
    bindings: dict[str, Any]
    references: frozenset[str]
    answer_entities: frozenset[str]
    preferences: dict[str, bool] = field(default_factory=dict)


class Grounder:
    """Graph operations and independent semantic handlers, keyed by stable family ID."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.obj = {o.symbol: o for o in scene.objects}
        self.room = {r.symbol: r for r in scene.rooms}
        self.place = {p.symbol: p for p in scene.places}
        self.oc = {k: {o.symbol for o in v} for k, v in scene.objects_by_class.items()}
        self.rc = {k: {r.symbol for r in v} for k, v in scene.rooms_by_class.items()}
        self.ro = {r: set(v) for r, v in scene.objects_by_room.items()}
        self.op = {
            o: set(scene.entity(o).parents) & self.place.keys() for o in self.obj
        }
        self.place_rooms = {p: set(scene.rooms_for_entity(p)) for p in self.place}
        self.oroom = {
            o: {r for r, contents in self.ro.items() if o in contents} for o in self.obj
        }
        self.ra = scene.adjacency("room")
        self.pa = scene.adjacency("place")
        self.layers = {
            kind
            for kind in ("mesh_place", "place")
            if any(p.kind == kind for p in scene.places)
        }
        self._distance: dict[tuple[str, str], float] = {}
        self._paths: dict[tuple[str, str, tuple[str, ...]], dict[str, list[str]]] = {}
        self.resolved: set[str] = set()
        self.binding: dict[str, Any] = {}
        self.preferences: dict[str, bool] = {}
        self._hop_distances: dict[tuple, dict[str, int]] = {}
        self._route_costs: dict[str, tuple[dict, dict]] = {}

    def prefer(self, condition: Any, name: str) -> None:
        """Record a difficulty preference without rejecting a valid answer."""
        self.preferences[name] = self.preferences.get(name, True) and bool(condition)

    def hop_distances(self, layer, start):
        key = layer, start
        if key not in self._hop_distances:
            adjacency = self.ra if layer == "room" else self.pa
            distances = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in sorted_symbols(adjacency.get(node, set())):
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            self._hop_distances[key] = distances
        return self._hop_distances[key]

    def distance(self, a: str, b: str) -> float:
        key = tuple(sorted((a, b)))
        if key not in self._distance:
            self._distance[key] = math.dist(
                self.scene.entity(a).position, self.scene.entity(b).position
            )
        return self._distance[key]

    def paths(
        self, layer: str, start: str, forbidden: Iterable[str] = ()
    ) -> dict[str, list[str]]:
        banned = tuple(sorted(forbidden))
        key = layer, start, banned
        if key not in self._paths:
            adjacency = self.ra if layer == "room" else self.pa
            result = {} if start in banned else {start: [start]}
            queue = deque(result)
            while queue:
                node = queue.popleft()
                for neighbor in sorted_symbols(
                    adjacency.get(node, set()) - set(banned)
                ):
                    if neighbor not in result:
                        result[neighbor] = result[node] + [neighbor]
                        queue.append(neighbor)
            self._paths[key] = result
        return self._paths[key]

    def components(self, forbidden: Iterable[str] = ()) -> list[set[str]]:
        remaining = self.room.keys() - set(forbidden)
        result = []
        while remaining:
            component = set(self.paths("room", sorted_symbols(remaining)[0], forbidden))
            result.append(component)
            remaining -= component
        return result

    def extrema(self, scores: dict[Any, Any], maximum: bool = False) -> set:
        require(scores, "no comparison candidates")
        best = (max if maximum else min)(scores.values())
        return {
            k
            for k, v in scores.items()
            if v == best
            or (
                isinstance(v, float)
                and math.isclose(v, best, rel_tol=0, abs_tol=TIE_TOLERANCE)
            )
        }

    def unique(self, values: Iterable[str], reason: str) -> str:
        values = set(values)
        require(len(values) == 1, reason)
        value = next(iter(values))
        self.resolved.add(value)
        return value

    def objects(self, role: str = "target", room_role: str | None = None) -> set[str]:
        result = self.oc.get(self.binding.get(role + "_object_class"), set())
        if room_role:
            result = result & set().union(*(self.ro[r] for r in self.rooms(room_role)))
        return result

    def rooms(self, role: str = "source") -> set[str]:
        return self.rc.get(self.binding.get(role + "_room_class"), set())

    def reference(self, role: str = "reference", room_role: str | None = None) -> str:
        if room_role:
            self.prefer(
                self.objects(role) - self.objects(role, room_role),
                f"{role}_scope_has_outside_distractor",
            )
        return self.unique(
            self.objects(role, room_role),
            f"{role} reference is not unique in its stated scope",
        )

    def parent(self, obj: str, layer: str | None = None) -> str:
        parents = self.op[obj]
        if layer:
            parents = {p for p in parents if self.place[p].kind == layer}
        return self.unique(
            parents, "reference has no unique direct parent in the selected layer"
        )

    def anchor_room(self, role: str = "reference", room_role: str | None = None) -> str:
        obj = self.reference(role, room_role)
        rooms = set(self.oroom[obj])
        if room_role:
            rooms &= self.rooms(room_role)
        return self.unique(rooms, "reference has no unique containing room")

    def nearest_scores(
        self, targets: set[str], references: set[str], maximum: bool = False
    ) -> dict[str, float]:
        require(references, "empty reference class")
        return {
            o: (max if maximum else min)(self.distance(o, a) for a in references - {o})
            for o in targets
            if references - {o}
        }

    def content_rooms(
        self, targets: set[str], allowed: set[str] | None = None
    ) -> set[str]:
        return {
            r
            for r in (self.room.keys() if allowed is None else allowed)
            if self.ro[r] & targets
        }

    def scoped_targets(self) -> set[str]:
        targets = self.objects(room_role="source")
        require(len(targets) >= 2, "fewer than two scoped target candidates")
        require(self.objects() - targets, "room-category filter removes no target")
        return targets

    def room_groups(self) -> dict[str, set[str]]:
        groups = {r: self.ro[r] & self.objects() for r in self.rooms()}
        return groups

    def eligible_groups(self) -> dict[str, tuple[set[str], set[str]]]:
        groups = {
            r: (self.ro[r] & self.objects(), self.ro[r] & self.objects("reference"))
            for r in self.rooms()
        }
        return {r: (t, u) for r, (t, u) in groups.items() if t and u}

    def hop_targets(self) -> set[str]:
        b = self.binding
        ref = self.reference(room_role="source")
        start = self.parent(ref, b["place_layer_name"])
        paths = self.paths("place", start)
        hops = b["hop_limit"]
        selected = {p for p, path in paths.items() if len(path) - 1 <= hops}
        targets = self.objects() - {ref}
        result = {o for o in targets if self.op[o] & selected}
        require(
            len(result) >= 2 and targets - result,
            "hop filter needs selected and excluded targets",
        )
        require(
            any(
                any(len(paths[p]) - 1 > 1 for p in self.op[o] & selected)
                for o in result
            ),
            "one hop already covers all targets",
        )
        return result

    def mutual_pairs(self) -> set[tuple[str, str]]:
        t, u = self.objects(), self.objects("reference")
        require(
            len(t) >= 3 and len(u) >= 3,
            "mutual nearest needs competing members of both classes",
        )
        forward = {o: self.extrema({a: self.distance(o, a) for a in u}) for o in t}
        reverse = {a: self.extrema({o: self.distance(o, a) for o in t}) for a in u}
        result = {(o, a) for o in t for a in forward[o] if o in reverse[a]}
        require(
            len(result) >= 2 and len(result) < sum(map(len, forward.values())),
            "no contrast with one-way or globally nearest matches",
        )
        return result

    def geometric_pairs(
        self, same_room: bool = False, different_rooms: bool = False
    ) -> set[tuple[str, str]]:
        t, u = self.objects(), self.objects("reference")
        all_scores = {(o, a): self.distance(o, a) for o in t for a in u if o != a}
        scores = dict(all_scores)
        if same_room:
            scores = {
                pair: d
                for pair, d in scores.items()
                if self.oroom[pair[0]] & self.oroom[pair[1]] & self.rooms()
            }
        if different_rooms:
            scores = {
                pair: d
                for pair, d in scores.items()
                if self.oroom[pair[0]]
                and self.oroom[pair[1]]
                and not self.oroom[pair[0]] & self.oroom[pair[1]]
            }
        require(len(scores) >= 3, "fewer than three eligible pairs")
        result = self.extrema(scores)
        if same_room or different_rooms:
            require(
                result != self.extrema(all_scores),
                "pair containment filter does not change the optimum",
            )
        return result

    def evaluate(self, entry: dict, bindings: dict) -> Candidate:
        self.binding = bindings
        self.preferences = {}
        self.resolved = {
            v
            for v in bindings.values()
            if isinstance(v, str) and v in self.scene.entities
        }
        family, suffix = entry["id"].split("_", 1)
        handler = getattr(self, family + "_" + suffix, None)
        if handler:
            solution = handler()
        else:
            solution = self.legacy(family, suffix)
        require(
            isinstance(solution, str) and solution not in {"", "<>", "{}"},
            "empty positive answer",
        )
        if family == "pddl":
            atoms = re.findall(r"\((?!and\b|or\b|not\b)[a-z-]+\s[^()]+\)", solution)
            require(len(atoms) <= 40, "goal exceeds 40 fact occurrences")
            # Count operands of each OR without distributing nested formulas.
            stack = []
            operands = 0
            for token in re.findall(r"\(|\)|[^\s()]+", solution):
                if token == "(":
                    if stack and stack[-1][0] == "or":
                        operands += 1
                    stack.append([None])
                elif token == ")":
                    stack.pop()
                elif stack and stack[-1][0] is None:
                    stack[-1][0] = token
            require(operands <= 30, "goal exceeds 30 explicit OR operands")
        params = {}
        for name, typ in entry["parameters"].items():
            value = bindings[parameter_role(name, typ)]
            if "_class_" in typ:
                value = noun_form(
                    self.scene, value, typ, bindings.get("required_count")
                )
            elif typ == "distance_m":
                value = format_number(value)
            elif typ == "place_layer_name":
                value = {"mesh_place": "2D", "place": "3D"}[value]
            params[name] = value
        tokens = set(re.findall(r"[A-Za-z_]\w*", solution))
        return Candidate(
            Question(params, solution),
            dict(bindings),
            frozenset(self.resolved),
            frozenset(tokens & self.scene.entities.keys()),
            dict(self.preferences),
        )

    # QA semantic handlers. Shared helpers enforce distinct identity and scope.
    def qa_class_count_in_room_class(self):
        return str(len(self.scoped_targets()))

    def qa_class_near_class(self):
        t = self.objects()
        scores = self.nearest_scores(t, self.objects("reference"))
        require(
            len(self.objects("reference")) >= 2 and len(t) >= 3,
            "insufficient spatial candidates",
        )
        selected = {o for o, d in scores.items() if d <= self.binding["radius_m"]}
        require(selected and t - selected, "radius does not divide targets")
        return set_answer(selected)

    def qa_closest_class_pairs_in_room_class(self):
        return pair_answer(self.geometric_pairs(same_room=True))

    def qa_counts_per_room(self):
        counts = {r: len(t) for r, t in self.room_groups().items()}
        require(
            len(counts) >= 2 and len(set(counts.values())) >= 2,
            "need multiple rooms with varying counts",
        )
        self.prefer(0 in counts.values(), "includes_zero_count")
        self.prefer(len(set(counts.values()) - {0}) >= 2, "unequal_positive_counts")
        return dictionary(counts)

    def qa_rooms_with_both_classes(self):
        t, u = self.objects(), self.objects("reference")
        rooms = self.rooms()
        a, b = self.content_rooms(t, rooms), self.content_rooms(u, rooms)
        require(
            a & b and a - b and b - a, "no contrast with single-class room selection"
        )
        self.prefer(
            self.content_rooms(t) & self.content_rooms(u) - rooms,
            "room_category_excludes_matching_room",
        )
        return set_answer(a & b)

    def qa_rooms_with_target_without_excluded(self):
        a, b = (
            self.content_rooms(self.objects()),
            self.content_rooms(self.objects("excluded")),
        )
        require(a - b and a & b, "need rooms with target only and both classes")
        return set_answer(a - b)

    def qa_rooms_with_exact_count(self):
        counts = {r: len(t) for r, t in self.room_groups().items()}
        n = self.binding["required_count"]
        require(
            n >= 1 and n in counts.values() and any(v != n for v in counts.values()),
            "exact count needs matching and nonmatching rooms",
        )
        self.prefer(
            any(v < n for v in counts.values()) and any(v > n for v in counts.values()),
            "lower_and_higher_count_distractors",
        )
        return set_answer(r for r, v in counts.items() if v == n)

    def qa_classes_in_every_room(self):
        groups = [
            {self.obj[o].semantic_class for o in self.ro[r]} - {None}
            for r in self.rooms()
        ]
        require(len(groups) >= 3, "universal room coverage needs three rooms")
        result = set.intersection(*groups)
        require(
            result and all(result != group for group in groups),
            "intersection is empty or equals an individual inventory",
        )
        return class_answer(result)

    def qa_objects_sharing_place(self):
        t = self.objects()
        u = self.objects("reference", "source")
        result = {
            o
            for o in t
            for a in u
            if self.op[o] & self.op[a] and self.oroom[o] & self.oroom[a] & self.rooms()
        }
        same_room = {
            o for o in t for a in u if self.oroom[o] & self.oroom[a] & self.rooms()
        }
        require(
            result and same_room - result, "same-place restriction adds no contrast"
        )
        return set_answer(result)

    def qa_rooms_with_most_targets(self):
        groups = self.room_groups()
        counts = {r: len(t) for r, t in groups.items()}
        require(
            len(counts) >= 3 and len(set(counts.values())) > 1,
            "need competing per-room counts",
        )
        result = self.extrema(counts, True)
        require(
            result != self.extrema({r: len(self.ro[r]) for r in counts}, True),
            "total-count shortcut finds the same rooms",
        )
        return set_answer(result)

    def fraction_rooms(self, role="source"):
        scores = {
            r: Fraction(len(self.ro[r] & self.objects()), len(self.ro[r]))
            for r in self.rooms(role)
            if self.ro[r]
        }
        require(
            len(scores) >= 3 and len(set(scores.values())) > 1,
            "need three nonempty rooms with unequal proportions",
        )
        result = self.extrema(scores, True)
        require(
            result
            != self.extrema(
                {r: len(self.ro[r] & self.objects()) for r in scores}, True
            ),
            "fraction and count have the same winner",
        )
        return result

    def qa_rooms_with_highest_target_fraction(self):
        return set_answer(self.fraction_rooms())

    def inventories(self, role):
        return {
            self.obj[o].semantic_class for r in self.rooms(role) for o in self.ro[r]
        } - {None}

    def qa_classes_shared_between_room_classes(self):
        a, b = self.inventories("source"), self.inventories("comparison")
        require(a & b and a - b and b - a, "inventory intersection lacks contrast")
        return class_answer(a & b)

    def qa_classes_exclusive_between_room_classes(self):
        a, b = self.inventories("source"), self.inventories("comparison")
        require(
            a - b and a & b and len(self.rooms("comparison")) >= 2,
            "inventory difference lacks contrast or comparison rooms",
        )
        return class_answer(a - b)

    def qa_place_counts_in_content_selected_room(self):
        room = self.anchor_room(room_role="source")
        require(
            len(self.rooms()) >= 2
            and self.objects("reference") - self.objects("reference", "source"),
            "room reference needs disambiguation",
        )
        places = self.scene.places_by_room[room]
        require(len(places) >= 2, "fewer than two place children")
        return str(len(places))

    def qa_targets_nearer_external_reference(self):
        targets, references = self.objects(), self.objects("reference")
        require(
            all(len(self.oroom[o]) <= 1 for o in targets),
            "target room membership is ambiguous",
        )
        eligible = set()
        result = set()
        for target in sorted_symbols(targets):
            local = {ref for ref in references if self.oroom[target] & self.oroom[ref]}
            if not local:
                continue
            eligible.add(target)
            external = {
                ref
                for ref in references
                if self.oroom[ref] and not self.oroom[target] & self.oroom[ref]
            }
            if external and min(
                self.distance(target, ref) for ref in external
            ) + TIE_TOLERANCE < min(self.distance(target, ref) for ref in local):
                result.add(target)
        require(
            result and eligible - result,
            "need targets with closer external and closer-or-tied local references",
        )
        return set_answer(result)

    def nearest_per_room(self, top_k=1):
        groups = self.eligible_groups()
        require(groups, "no rooms contain both classes")
        result = {}
        contrast = False
        competing = 0
        for r, (t, u) in groups.items():
            require(len(t) >= top_k, "fewer targets than the requested rank")
            competing += len(t) > top_k
            scores = self.nearest_scores(t, u)
            cutoff = sorted(scores.values())[top_k - 1]
            result[r] = {o for o, d in scores.items() if d <= cutoff + TIE_TOLERANCE}
            global_scores = self.nearest_scores(t, self.objects("reference"))
            contrast |= self.extrema(global_scores) != self.extrema(scores)
        require(competing, "ranking has no competing targets")
        self.prefer(len(groups) >= 2, "multiple_room_groups")
        self.prefer(competing >= 2, "multiple_competing_groups")
        self.prefer(contrast, "room_local_reference_changes_optimum")
        return result

    def qa_nearest_targets_per_room(self):
        return dictionary(
            {r: set_answer(t) for r, t in self.nearest_per_room().items()}
        )

    def qa_nearest_to_room_disambiguated_reference(self):
        ref = self.reference(room_role="source")
        t = self.objects()
        require(
            len(t) >= 3 and len(self.objects("reference")) >= 2,
            "insufficient reference disambiguation or targets",
        )
        result = self.extrema({o: self.distance(o, ref) for o in t})
        require(
            all(not self.oroom[o] & self.rooms() for o in result),
            "global minimum is already local",
        )
        require(
            result != self.extrema(self.nearest_scores(t, self.objects("reference"))),
            "arbitrary references do not change the answer",
        )
        return set_answer(result)

    def qa_global_farthest_from_selected_room(self):
        r = self.anchor_room(room_role="source")
        t = self.objects()
        require(
            len(t) >= 3 and t & self.ro[r] and t - self.ro[r],
            "need targets inside and outside the room",
        )
        result = self.extrema({o: self.distance(o, r) for o in t}, True)
        require(not result & self.ro[r], "farthest target is already local")
        require(
            len(self.rooms()) >= 2 and len(self.objects("reference")) >= 2,
            "room reference needs disambiguation",
        )
        return set_answer(result)

    def radius_all(self, far=False):
        t, u = self.objects(), self.objects("reference")
        radius = self.binding["radius_m"]
        require(len(t) >= 3 and len(u) >= 2, "need competing targets and references")
        nearby = {o: {a for a in u if self.distance(o, a) <= radius} for o in t}
        result = (
            {o for o, a in nearby.items() if not a}
            if far
            else {o for o, a in nearby.items() if a == u}
        )
        require(
            result and any(0 < len(a) < len(u) for a in nearby.values()),
            "universal condition has no partial-match contrast",
        )
        return result

    def qa_targets_near_every_reference(self):
        return set_answer(self.radius_all())

    def qa_targets_near_no_reference(self):
        return set_answer(self.radius_all(True))

    def qa_mean_nearest_reference_distance(self):
        t, u = self.objects(), self.objects("reference")
        require(len(t) >= 3 and len(u) >= 2, "insufficient mean-of-minima candidates")
        scores = self.nearest_scores(t, u)
        mean = sum(scores.values()) / len(scores)
        nearest_refs = {
            a for o in t for a in self.extrema({a: self.distance(o, a) for a in u})
        }
        require(
            len(nearest_refs) >= 2
            and max(scores.values()) - min(scores.values()) > 0.01,
            "nearest references or distances do not vary",
        )
        pair_mean = sum(self.distance(o, a) for o in t for a in u) / (len(t) * len(u))
        require(
            abs(mean - pair_mean) > 0.01 and abs(mean - min(scores.values())) > 0.01,
            "aggregation shortcuts give the same result",
        )
        return format_number(mean)

    def minimax(self):
        t, u = self.objects(), self.objects("reference")
        require(len(t) >= 3 and len(u) >= 2, "insufficient minimax candidates")
        result = self.extrema(self.nearest_scores(t, u, True))
        require(
            result != self.extrema(self.nearest_scores(t, u)),
            "minimax equals nearest-to-any",
        )
        require(
            result
            != self.extrema(
                {o: sum(self.distance(o, a) for a in u) / len(u) for o in t}
            ),
            "minimax equals minimum mean",
        )
        return result

    def qa_minimax_target(self):
        return set_answer(self.minimax())

    def qa_top_targets_per_room(self):
        require(self.binding["top_k"] >= 2, "ranking budget must exceed one")
        return dictionary(
            {
                r: set_answer(t)
                for r, t in self.nearest_per_room(self.binding["top_k"]).items()
            }
        )

    def qa_closest_pair_in_different_rooms(self):
        return pair_answer(self.geometric_pairs(different_rooms=True))

    def qa_targets_within_place_hops(self):
        return set_answer(self.hop_targets())

    def qa_target_count_within_place_hops(self):
        result = self.hop_targets()
        require(len(result) >= 3, "need three counted targets")
        return str(len(result))

    def endpoint_rooms(self):
        a, b = self.anchor_room(), self.anchor_room("destination")
        require(a != b, "route endpoints coincide")
        return a, b

    def qa_room_hops_between_content_anchors(self):
        a, b = self.endpoint_rooms()
        path = self.paths("room", a).get(b)
        require(path and len(path) >= 4, "need a connected nontrivial route")
        return str(len(path) - 1)

    def restricted_route(self):
        a, b = self.endpoint_rooms()
        banned = self.rooms("excluded")
        require(a not in banned and b not in banned, "route endpoint is forbidden")
        path = self.paths("room", a, banned).get(b)
        original = self.paths("room", a).get(b)
        require(path and original, "no route remains after excluding the category")
        self.prefer(len(path) > len(original), "positive_avoidance_detour")
        return a, b, path, original

    def qa_shortest_room_route_avoiding_class(self):
        a, b, path, _ = self.restricted_route()
        # The external comparator uses ordered list equality. Emit only uniquely
        # shortest instances until it supports path validity rather than strings.
        dist = {
            n: len(p) - 1
            for n, p in self.paths("room", a, self.rooms("excluded")).items()
        }
        counts = {a: 1}
        for node in sorted(dist, key=lambda n: dist[n]):
            for neighbor in self.ra[node] - self.rooms("excluded"):
                if dist.get(neighbor) == dist[node] + 1:
                    counts[neighbor] = min(
                        2, counts.get(neighbor, 0) + counts.get(node, 0)
                    )
        require(
            counts[b] == 1, "multiple shortest routes need a path-validity comparator"
        )
        return sldp_list(path)

    def qa_reachable_target_rooms_avoiding_class(self):
        start = self.anchor_room()
        banned = self.rooms("excluded")
        require(start not in banned, "starting room is forbidden")
        all_reached = set(self.paths("room", start))
        reached = set(self.paths("room", start, banned))
        targets = self.content_rooms(self.objects()) - {start}
        result = reached & targets
        require(
            result and (all_reached - banned - reached) & targets,
            "exclusion disconnects no allowed target room",
        )
        require(
            reached - targets - {start}, "target-content filter changes no reached room"
        )
        return set_answer(result)

    def qa_critical_rooms_between_anchors(self):
        a, b = self.endpoint_rooms()
        path = self.paths("room", a).get(b)
        require(path, "route endpoints are disconnected")
        result = {
            r for r in self.room.keys() - {a, b} if b not in self.paths("room", a, {r})
        }
        require(result, "no mandatory intermediate room")
        self.prefer(set(path[1:-1]) - result, "shortest_path_has_optional_intermediate")
        return set_answer(result)

    def qa_rooms_with_most_distinct_classes(self):
        scores = {
            r: len({self.obj[o].semantic_class for o in self.ro[r]} - {None})
            for r in self.rooms()
        }
        require(len(scores) >= 2, "need multiple rooms to compare diversity")
        self.prefer(len(set(scores.values())) > 1, "unequal_room_diversity")
        result = self.extrema(scores, True)
        self.prefer(
            result != self.extrema({r: len(self.ro[r]) for r in scores}, True),
            "diversity_differs_from_raw_count",
        )
        return set_answer(result)

    def qa_rooms_with_more_targets_than_references(self):
        counts = {
            r: (
                len(self.ro[r] & self.objects()),
                len(self.ro[r] & self.objects("reference")),
            )
            for r in self.rooms()
        }
        require(
            len(counts) >= 3
            and any(a == b for a, b in counts.values())
            and any(a < b for a, b in counts.values()),
            "need equal and reversed count contrasts",
        )
        return set_answer(r for r, (a, b) in counts.items() if a > b)

    def histogram(self):
        ids = set().union(*(self.ro[r] for r in self.rooms()))
        counts = Counter(
            self.obj[o].semantic_class for o in ids if self.obj[o].semantic_class
        )
        require(
            len(counts) >= 3 and len(set(counts.values())) > 1,
            "need three classes with unequal counts",
        )
        require(
            counts
            != Counter(o.semantic_class for o in self.obj.values() if o.semantic_class),
            "room-category filter leaves the histogram unchanged",
        )
        return counts

    def qa_class_histogram_in_room_class(self):
        return dictionary(dict(self.histogram()))

    def qa_rarest_classes_in_room_class(self):
        result = self.extrema(dict(self.histogram()))
        require(
            result != self.extrema({k: len(v) for k, v in self.oc.items()}),
            "local and global rarest classes coincide",
        )
        return class_answer(result)

    def qa_targets_in_distance_band(self):
        ref = self.reference()
        inner = self.binding["inner_radius_m"]
        outer = self.binding["outer_radius_m"]
        scores = {o: self.distance(o, ref) for o in self.objects()}
        require(
            0 < inner < outer
            and any(d <= inner for d in scores.values())
            and any(d > outer for d in scores.values()),
            "band needs candidates below and above",
        )
        return set_answer(o for o, d in scores.items() if inner < d <= outer)

    def closer_targets(self):
        a, b = self.reference(), self.reference("comparison")
        t = self.objects()
        selected = {o for o in t if self.distance(o, a) < self.distance(o, b)}
        require(
            len(selected) >= 2 and t - selected,
            "distance comparison needs selected and excluded targets",
        )
        return selected

    def qa_targets_closer_to_first_reference(self):
        return set_answer(self.closer_targets())

    def qa_targets_near_two_reference_classes(self):
        t = self.objects()
        radius = self.binding["radius_m"]
        a = {
            o
            for o, d in self.nearest_scores(t, self.objects("reference")).items()
            if d <= radius
        }
        b = {
            o
            for o, d in self.nearest_scores(t, self.objects("comparison")).items()
            if d <= radius
        }
        require(
            a & b and a - b and b - a and t - (a | b),
            "need targets near both, either, and neither reference class",
        )
        return set_answer(a & b)

    def qa_nearest_targets_outside_reference_room(self):
        a = self.reference()
        self.unique(self.oroom[a], "reference must have one containing room")
        t = self.objects()
        eligible = {o for o in t if not self.oroom[o] & self.oroom[a]}
        require(len(eligible) >= 2, "need multiple eligible targets")
        result = self.extrema({o: self.distance(o, a) for o in eligible})
        require(
            result != self.extrema({o: self.distance(o, a) for o in t}),
            "room exclusion does not change nearest targets",
        )
        return set_answer(result)

    def qa_mutual_nearest_class_pairs(self):
        return pair_answer(self.mutual_pairs())

    def qa_closest_class_pairs_per_room(self):
        groups = self.eligible_groups()
        require(groups, "no rooms contain both classes")
        result = {}
        competing = 0
        for r, (t, u) in groups.items():
            scores = {(o, a): self.distance(o, a) for o in t for a in u}
            competing += len(scores) > 1
            result[r] = pair_answer(self.extrema(scores))
        require(competing, "no competing within-room pairs")
        self.prefer(competing >= 2, "multiple_competing_groups")
        self.prefer(len(groups) >= 2, "multiple_room_groups")
        return dictionary(result)

    def qa_nearby_target_counts_per_reference(self):
        refs = self.objects("reference", "source")
        require(len(refs) >= 3, "need three scoped references")
        counts = {
            a: sum(
                self.distance(o, a) <= self.binding["radius_m"] for o in self.objects()
            )
            for a in refs
        }
        require(
            0 in counts.values() and len(set(counts.values())) > 1,
            "need varying counts including zero",
        )
        require(
            self.objects("reference") - refs,
            "room-category filter removes no reference",
        )
        return dictionary(counts)

    def qa_rooms_adjacent_to_two_room_classes(self):
        a = {r for r in self.room if self.ra[r] & self.rooms()}
        b = {r for r in self.room if self.ra[r] & self.rooms("comparison")}
        require(
            a & b and a - b and b - a,
            "adjacency intersection lacks single-category contrasts",
        )
        return set_answer(a & b)

    def qa_rooms_with_only_allowed_neighbors(self):
        allowed = self.rooms()
        result = {r for r in self.room if self.ra[r] and self.ra[r] <= allowed}
        require(
            result
            and any(self.ra[r] & allowed and self.ra[r] - allowed for r in self.room),
            "need qualifying and mixed-neighbor rooms",
        )
        return set_answer(result)

    def qa_target_rooms_within_room_hops(self):
        start = self.anchor_room()
        h = self.binding["hop_limit"]
        paths = self.paths("room", start)
        targets = self.content_rooms(self.objects()) - {start}
        result = {r for r in targets if r in paths and len(paths[r]) - 1 <= h}
        require(
            h >= 2
            and result
            and targets - result
            and any(len(paths[r]) > 2 for r in result),
            "bounded room-hop filter lacks contrast",
        )
        return set_answer(result)

    def qa_common_place_hop_neighborhood(self):
        layer = self.binding["place_layer_name"]
        h = self.binding["hop_limit"]
        a, b = (
            self.parent(self.reference(), layer),
            self.parent(self.reference("comparison"), layer),
        )
        require(a != b and h >= 2, "hop anchors coincide or budget is trivial")
        x = {p for p, path in self.paths("place", a).items() if len(path) - 1 <= h}
        y = {p for p, path in self.paths("place", b).items() if len(path) - 1 <= h}
        require(x & y and x - y and y - x, "hop intersection lacks contrast")
        return set_answer(x & y)

    def qa_minimum_length_place_route(self):
        layer = self.binding["place_layer_name"]
        a = self.parent(self.reference(), layer)
        b = self.parent(self.reference("destination"), layer)
        require(a != b, "route endpoints coincide")
        if a not in self._route_costs:
            costs = {a: 0.0}
            queue = [(0.0, a)]
            while queue:
                cost, node = heapq.heappop(queue)
                if cost > costs[node]:
                    continue
                for neighbor in sorted_symbols(self.pa[node]):
                    new = cost + self.distance(node, neighbor)
                    if new < costs.get(neighbor, math.inf):
                        costs[neighbor] = new
                        heapq.heappush(queue, (new, neighbor))
            # Minimum length among minimum-hop routes, on the BFS DAG.
            hops = self.hop_distances("place", a)
            hop_cost = {a: 0.0}
            for node in sorted(hops, key=lambda n: (hops[n], n)):
                for neighbor in sorted_symbols(self.pa[node]):
                    if hops[neighbor] == hops[node] + 1:
                        hop_cost[neighbor] = min(
                            hop_cost.get(neighbor, math.inf),
                            hop_cost[node] + self.distance(node, neighbor),
                        )
            self._route_costs[a] = costs, hop_cost
        costs, hop_cost = self._route_costs[a]
        require(b in costs, "place endpoints are disconnected")
        self.prefer(
            costs[b] + 0.01 < hop_cost[b],
            "weighted_differs_from_hop_optimum",
        )
        return format_number(costs[b])

    def qa_room_avoidance_detour(self):
        _, _, path, original = self.restricted_route()
        return str(len(path) - len(original))

    def qa_room_components_after_category_removal(self):
        banned = self.rooms("excluded")
        result = self.components(banned)
        require(
            len(self.room) - len(banned) >= 3
            and len(result) != len(self.components())
            and any(len(c) > 1 for c in result),
            "category removal does not change nontrivial components",
        )
        return str(len(result))

    def qa_rooms_with_all_targets_near_references(self):
        radius = self.binding["radius_m"]
        selected = set()
        partial = False
        for r in self.rooms():
            t = self.ro[r] & self.objects()
            u = self.ro[r] & self.objects("reference")
            good = {o for o in t if any(self.distance(o, a) <= radius for a in u)}
            if t and good == t:
                selected.add(r)
            partial |= bool(good and t - good)
        require(selected and partial, "need fully and partially covered target rooms")
        return set_answer(selected)

    def qa_highest_target_in_selected_room(self):
        r = self.anchor_room()
        t = self.objects() & self.ro[r]
        require(len(t) >= 2, "need competing scoped height candidates")
        scores = {o: self.height(o) for o in t}
        result = self.extrema(scores, True)
        self.prefer(len(set(scores.values())) > 1, "different_local_heights")
        self.prefer(
            any(self.height(o) > max(scores.values()) for o in self.objects() - t),
            "higher_target_outside_room",
        )
        return set_answer(result)

    def height(self, symbol):
        axis = self.scene._metadata.get("question_generation", {}).get("up_axis", "z")
        require(axis in {"x", "y", "z"}, "unknown upward axis")
        return self.scene.entity(symbol).position["xyz".index(axis)]

    def answer_ids(self, answer: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_]\w*", answer)) & self.scene.entities.keys()

    # PDDL handlers compute grounded Boolean goals; they do not replace the
    # validator supplied by heracles_agents.
    def pddl_visit_any_class_in_room_class(self):
        return facts("visited-object", self.scoped_targets())

    def pddl_inspect_closest_between_classes(self):
        t, u = self.objects(), self.objects("reference")
        require(len(t) >= 3 and len(u) >= 2, "insufficient spatial alternatives")
        result = self.extrema(self.nearest_scores(t, u))
        require(
            any(self.extrema({o: self.distance(o, a) for o in t}) != result for a in u),
            "reference choice does not affect grounding",
        )
        return facts("safe", result)

    def pddl_inspect_every_target_in_room_class(self):
        return facts("safe", self.scoped_targets(), "and")

    def pddl_visit_rooms_with_both_classes(self):
        a = self.content_rooms(self.objects())
        b = self.content_rooms(self.objects("reference"))
        require(
            len(a & b) >= 2 and (a - b or b - a),
            "room-content conjunction lacks contrast",
        )
        return facts("visited-region", a & b, "and")

    def pddl_end_in_max_count_room(self):
        return facts("in-region", self.answer_ids(self.qa_rooms_with_most_targets()))

    def per_room_targets(self):
        groups = {r: t for r, t in self.room_groups().items() if t}
        require(len(groups) >= 2, "need targets in multiple rooms")
        self.prefer(
            all(len(t) >= 2 for t in groups.values()), "multiple_choices_in_every_room"
        )
        require(
            not set.intersection(*groups.values()), "one target satisfies every room"
        )
        self.prefer(
            self.objects() - set().union(*groups.values()),
            "target_outside_room_category",
        )
        return groups

    def pddl_visit_one_target_per_room(self):
        return fold(
            "and",
            (
                facts("visited-object", t)
                for _, t in sorted(self.per_room_targets().items())
            ),
        )

    def pddl_inspect_and_hold_same_choice(self):
        return fold(
            "or",
            (
                fold("and", [pddl_fact("safe", o), pddl_fact("holding", o)])
                for o in sorted_symbols(self.scoped_targets())
            ),
        )

    def pddl_inspect_choice_end_in_same_room(self):
        t = self.scoped_targets()
        bindings = {(o, r) for o in t for r in self.oroom[o] & self.rooms()}
        require(
            len({r for _, r in bindings}) >= 2
            and len(bindings) < len(t) * len({r for _, r in bindings}),
            "object-room choices are independent or singular",
        )
        return fold(
            "or",
            (
                fold("and", [pddl_fact("safe", o), pddl_fact("in-region", r)])
                for o, r in sorted(bindings)
            ),
        )

    def pddl_visit_all_near_reference_class(self):
        selected = self.answer_ids(self.qa_class_near_class())
        require(len(selected) >= 3, "need three universal visit targets")
        return facts("visited-object", selected, "and")

    def pddl_inspect_near_target_far_from_excluded(self):
        t = self.objects()
        radius = self.binding["radius_m"]
        near = {
            o
            for o, d in self.nearest_scores(t, self.objects("reference")).items()
            if d <= radius
        }
        far = {
            o
            for o, d in self.nearest_scores(t, self.objects("excluded")).items()
            if d > radius
        }
        require(
            len(near & far) >= 2 and near - far and far - near,
            "spatial inclusion/exclusion lacks independent effects",
        )
        return facts("safe", near & far, "and")

    def pddl_move_unique_target_to_reference_place(self):
        o = self.reference("target", "source")
        a = self.reference("reference", "destination")
        p = self.parent(a)
        self.prefer(
            self.objects() - {o} and self.objects("reference") - {a},
            "scoped_references_have_outside_distractors",
        )
        require(p not in self.op[o], "relocation is already satisfied")
        return pddl_fact("object-in-place", o, p)

    def pddl_move_any_target_beside_selected_reference(self):
        t = self.objects()
        u = self.objects("reference", "destination")
        require(
            len(t) >= 2 and len(u) >= 2 and self.objects("reference") - u,
            "insufficient source/destination distractors",
        )
        parents = {a: self.parent(a) for a in u}
        bindings = {
            (o, parents[a])
            for o in t
            for a in u
            if self.oroom[o]
            and not self.oroom[o] & self.oroom[a]
            and parents[a] not in self.op[o]
        }
        require(
            len({p for _, p in bindings}) >= 2
            and len(bindings) < len(t) * len(set(parents.values())),
            "source-destination correlation is trivial",
        )
        return fold(
            "or", (pddl_fact("object-in-place", o, p) for o, p in sorted(bindings))
        )

    def relocation_sets(self):
        t = self.scoped_targets()
        u = self.objects("reference", "destination")
        d = {self.parent(a) for a in u}
        require(
            len(d) >= 2 and self.objects("reference") - u,
            "destination scope needs alternatives and distractors",
        )
        require(
            all(not self.op[o] & d for o in t),
            "a relocation alternative is already satisfied",
        )
        return t, d

    def pddl_move_all_to_one_shared_destination(self):
        t, d = self.relocation_sets()
        return fold(
            "or",
            (
                fold(
                    "and",
                    (pddl_fact("object-in-place", o, p) for o in sorted_symbols(t)),
                )
                for p in sorted_symbols(d)
            ),
        )

    def pddl_move_each_to_independent_destination(self):
        t, d = self.relocation_sets()
        return fold(
            "and",
            (
                fold(
                    "or",
                    (pddl_fact("object-in-place", o, p) for p in sorted_symbols(d)),
                )
                for o in sorted_symbols(t)
            ),
        )

    def pddl_move_each_to_nearest_reference_place(self):
        t = self.scoped_targets()
        u = self.objects("reference")
        require(len(u) >= 3, "need competing reference destinations")
        clauses = []
        destinations = set()
        for o in sorted_symbols(t):
            refs = self.extrema({a: self.distance(o, a) for a in u})
            d = {self.parent(a) for a in refs}
            require(not d & self.op[o], "nearest relocation is already satisfied")
            destinations.update(d)
            clauses.append(
                fold(
                    "or",
                    (pddl_fact("object-in-place", o, p) for p in sorted_symbols(d)),
                )
            )
        require(len(destinations) >= 2, "all objects have the same nearest destination")
        return fold("and", clauses)

    def pddl_inspect_nearest_target_per_room(self):
        return fold(
            "and",
            (facts("safe", t) for _, t in sorted(self.nearest_per_room().items())),
        )

    def pddl_visit_target_rooms_avoid_excluded_class(self):
        r = self.content_rooms(self.objects())
        f = self.rooms("excluded")
        require(
            len(r) >= 2 and f and not r & f,
            "positive/negative room sets conflict or are empty",
        )
        component = set(self.paths("room", sorted_symbols(r)[0], f))
        require(
            r <= component and component - r,
            "target rooms lack an allowed connecting component or content contrast",
        )
        return fold(
            "and",
            [
                facts("visited-region", r, "and"),
                fold("and", (negated("visited-region", x) for x in sorted_symbols(f))),
            ],
        )

    def pddl_end_in_room_without_excluded_object(self):
        a = self.content_rooms(self.objects(), self.rooms("destination"))
        f = self.content_rooms(self.objects("excluded"))
        require(
            len(a - f) >= 2 and a & f,
            "negative content filter lacks candidates and contrast",
        )
        require(
            self.content_rooms(self.objects()) - f - self.rooms("destination"),
            "room-category filter has no effect",
        )
        return facts("in-region", a - f)

    def pddl_finish_holding_no_scoped_target(self):
        return fold(
            "and",
            (negated("holding", o) for o in sorted_symbols(self.scoped_targets())),
        )

    def pddl_visit_choice_leave_same_object_unheld(self):
        return fold(
            "or",
            (
                fold("and", [pddl_fact("visited-object", o), negated("holding", o)])
                for o in sorted_symbols(self.scoped_targets())
            ),
        )

    def pddl_alternative_universal_tasks(self):
        t = self.scoped_targets()
        u = self.objects("reference", "destination")
        require(
            len(u) >= 2 and self.objects("reference") - u,
            "alternative scope has no candidates or effect",
        )
        return fold("or", [facts("safe", t, "and"), facts("visited-object", u, "and")])

    def pddl_inspections_with_destination_alternative(self):
        t = self.scoped_targets()
        r = self.rooms("destination")
        u = self.objects("reference")
        require(
            len(r) >= 2 and len(u) >= 2 and any(not self.oroom[o] & r for o in u),
            "destination alternatives lack independent candidates",
        )
        return fold(
            "and",
            [
                facts("safe", t, "and"),
                fold("or", [facts("in-region", r), facts("at-object", u)]),
            ],
        )

    def pddl_choose_room_inspect_all_its_targets(self):
        groups = self.per_room_targets()
        return fold(
            "or",
            (
                fold("and", [facts("safe", t, "and"), pddl_fact("in-region", r)])
                for r, t in sorted(groups.items())
            ),
        )

    def pddl_visit_target_rooms_without_excluded(self):
        positive = self.content_rooms(self.objects(), self.rooms())
        excluded = self.content_rooms(self.objects("excluded"), self.rooms())
        result = positive - excluded
        require(
            len(result) >= 2 and positive & excluded,
            "need multiple target-only rooms and an excluded target-containing room",
        )
        self.prefer(self.rooms() - positive, "room_without_target_distractor")
        return facts("visited-region", result, "and")

    def pddl_inspect_hop_reachable_targets(self):
        return facts("safe", self.hop_targets(), "and")

    def minimum_hop_targets(self):
        layer = self.binding["place_layer_name"]
        start = self.parent(self.reference(room_role="source"), layer)
        paths = self.paths("place", start)
        scores = {
            o: min(len(paths[p]) - 1 for p in self.op[o] if p in paths)
            for o in self.objects()
            if self.op[o] & paths.keys()
        }
        require(
            len(scores) >= 3
            and len(set(scores.values())) > 1
            and min(scores.values()) >= 2,
            "need nontrivial competing navigation distances",
        )
        result = self.extrema(scores)
        geometric = self.extrema(
            {o: self.distance(o, self.reference(room_role="source")) for o in scores}
        )
        require(result != geometric, "Euclidean-nearest target is also hop-optimal")
        return result, start, paths

    def pddl_end_at_minimum_hop_reference_place(self):
        layer = self.binding["place_layer_name"]
        start = self.parent(self.reference(room_role="source"), layer)
        paths = self.paths("place", start)
        d = {p for o in self.objects() for p in self.op[o] if p in paths}
        require(len(d) >= 3, "need three reachable destination locations")
        scores = {p: len(paths[p]) - 1 for p in d}
        require(min(scores.values()) >= 2, "minimum route is trivial")
        result = self.extrema(scores)
        require(
            result != self.extrema({p: self.distance(start, p) for p in d}),
            "geometric and hop destinations coincide",
        )
        return facts("at-place", result)

    def pddl_hold_minimum_hop_target(self):
        require(
            all(
                len(
                    {
                        p
                        for p in self.op[o]
                        if self.place[p].kind == self.binding["place_layer_name"]
                    }
                )
                == 1
                for o in self.objects()
            ),
            "target location is ambiguous",
        )
        return facts("holding", self.minimum_hop_targets()[0])

    def pddl_inspect_minimax_target(self):
        return facts("safe", self.minimax())

    def pddl_move_one_target_per_source_room(self):
        groups = self.per_room_targets()
        a = self.reference("reference", "destination")
        p = self.parent(a)
        require(
            self.objects("reference") - {a},
            "destination reference needs outside distractors",
        )
        require(
            all(p not in self.op[o] for t in groups.values() for o in t),
            "relocation is already satisfied",
        )
        return fold(
            "and",
            (
                fold(
                    "or",
                    (pddl_fact("object-in-place", o, p) for o in sorted_symbols(t)),
                )
                for _, t in sorted(groups.items())
            ),
        )

    def pddl_inspect_at_least_two_targets(self):
        t = self.scoped_targets()
        require(3 <= len(t) <= 5, "cardinality task needs three to five targets")
        return fold(
            "or",
            (facts("safe", pair, "and") for pair in combinations(sorted_symbols(t), 2)),
        )

    def pddl_inspect_targets_above_reference(self):
        a = self.reference()
        t = self.objects()
        result = {o for o in t if self.height(o) > self.height(a)}
        require(
            len(result) >= 2 and t - result,
            "height filter lacks selected/excluded targets",
        )
        return facts("safe", result, "and")

    def pddl_hold_target_closer_to_first_reference(self):
        return facts("holding", self.closer_targets())

    def pddl_inspect_mutual_nearest_pair(self):
        return fold(
            "or", (facts("safe", pair, "and") for pair in sorted(self.mutual_pairs()))
        )

    def pddl_inspect_closest_pair_in_room_class(self):
        return fold(
            "or",
            (
                facts("safe", pair, "and")
                for pair in sorted(self.geometric_pairs(same_room=True))
            ),
        )

    def pddl_end_in_highest_target_fraction_room(self):
        return facts("in-region", self.fraction_rooms("destination"))

    def pddl_visit_rooms_with_exact_target_count(self):
        result = self.answer_ids(self.qa_rooms_with_exact_count())
        self.prefer(len(result) >= 2, "multiple_exact_count_rooms")
        return facts("visited-region", result, "and")

    def pddl_visit_adjacent_rooms_without_target(self):
        adjacent = {r for r in self.room if self.ra[r] & self.rooms()}
        excluded = self.content_rooms(self.objects("excluded"))
        result = adjacent - excluded
        require(
            len(result) >= 2
            and adjacent & excluded
            and self.room.keys() - adjacent - excluded,
            "adjacency/content filters lack contrast",
        )
        return facts("visited-region", result, "and")

    def pddl_visit_connected_room_pair(self):
        a, b = self.rooms(), self.rooms("destination")
        pairs = {(r, s) for r in a for s in b if s in self.ra[r]}
        require(
            len(a) >= 2 and len(b) >= 2 and 2 <= len(pairs) < len(a) * len(b),
            "room choices lack nontrivial connectivity correlation",
        )
        return fold(
            "or",
            (
                fold("and", [pddl_fact("visited-region", r), pddl_fact("in-region", s)])
                for r, s in sorted(pairs)
            ),
        )

    def pddl_inspect_one_target_in_each_room_category(self):
        a = self.objects(room_role="source")
        b = self.objects(room_role="destination")
        require(
            len(a) >= 2 and len(b) >= 2 and not a & b and self.objects() - a - b,
            "need separate scoped choices and outside targets",
        )
        return fold("and", [facts("safe", a), facts("safe", b)])

    def pddl_hold_either_class_in_room_class(self):
        a = self.objects(room_role="source")
        b = self.objects("alternative", "source")
        require(
            len(a) >= 2
            and len(b) >= 2
            and self.objects() - a
            and self.objects("alternative") - b,
            "class alternatives need scoped choices and distractors",
        )
        return facts("holding", a | b)

    def pddl_inspect_all_or_hold_and_end_in_same_room(self):
        t = self.scoped_targets()
        u = self.objects("reference", "destination")
        pairs = {(o, r) for o in u for r in self.oroom[o] & self.rooms("destination")}
        rooms = {r for _, r in pairs}
        require(
            len(rooms) >= 2 and len(pairs) < len(u) * len(rooms),
            "compound alternative lacks object-room correlation",
        )
        return fold(
            "or",
            [
                facts("safe", t, "and"),
                fold(
                    "or",
                    (
                        fold(
                            "and", [pddl_fact("holding", o), pddl_fact("in-region", r)]
                        )
                        for o, r in sorted(pairs)
                    ),
                ),
            ],
        )

    def pddl_keep_target_out_of_reference_places(self):
        o = self.reference("target")
        d = {self.parent(a) for a in self.objects("reference")}
        require(
            len(d) >= 2 and self.op[o] & d,
            "forbidden places are singular or already avoided",
        )
        return fold(
            "and", (negated("object-in-place", o, p) for p in sorted_symbols(d))
        )

    def pddl_inspect_choice_without_holding_excluded(self):
        t = self.scoped_targets()
        u = self.objects("excluded")
        require(len(u) >= 2, "need multiple holding exclusions")
        return fold(
            "and",
            [
                facts("safe", t),
                fold("and", (negated("holding", o) for o in sorted_symbols(u))),
            ],
        )

    def pddl_end_adjacent_to_reference_avoid_category(self):
        start = self.anchor_room()
        f = self.rooms("excluded")
        d = self.rooms("destination") & self.ra[start] - f
        require(d and f, "no allowed adjacent destination or forbidden category")
        self.prefer(len(d) >= 2, "multiple_adjacent_destinations")
        self.prefer(self.rooms("destination") - d, "nonadjacent_destination_distractor")
        return fold(
            "and",
            [
                facts("in-region", d),
                fold("and", (negated("visited-region", r) for r in sorted_symbols(f))),
            ],
        )

    def pddl_visit_nearby_locations_except_start(self):
        layer = self.binding["place_layer_name"]
        start = self.parent(self.reference(), layer)
        paths = self.paths("place", start)
        h = self.binding["hop_limit"]
        selected = {p for p, path in paths.items() if 1 <= len(path) - 1 <= h}
        require(
            h >= 2
            and len(selected) >= 3
            and any(len(path) - 1 > h for path in paths.values()),
            "neighborhood has no bounded contrast",
        )
        return facts("visited-place", selected, "and")

    def pddl_visit_one_per_room_end_at_reference(self):
        return fold(
            "and",
            [
                self.pddl_visit_one_target_per_room(),
                pddl_fact("at-object", self.reference()),
            ],
        )

    def pddl_choose_room_choose_complete_task(self):
        groups = self.eligible_groups()
        require(len(groups) >= 2, "need multiple rooms containing both classes")
        self.prefer(
            all(len(t) >= 2 and len(u) >= 2 for t, u in groups.values()),
            "compound_alternatives_in_every_room",
        )
        return fold(
            "or",
            (
                fold("or", [facts("safe", t, "and"), facts("visited-object", u, "and")])
                for _, (t, u) in sorted(groups.items())
            ),
        )

    def pddl_move_target_to_minimum_hop_reference(self):
        o = self.reference("target")
        layer = self.binding["place_layer_name"]
        start = self.parent(o, layer)
        paths = self.paths("place", start)
        d = {self.parent(a, layer) for a in self.objects("reference")}
        d &= paths.keys()
        require(len(d) >= 3, "need three reachable reference destinations")
        scores = {p: len(paths[p]) - 1 for p in d}
        require(min(scores.values()) > 0, "destination is already occupied")
        result = self.extrema(scores)
        require(
            result != self.extrema({p: self.distance(start, p) for p in d}),
            "Euclidean and hop destinations coincide",
        )
        return fold(
            "or", (pddl_fact("object-in-place", o, p) for p in sorted_symbols(result))
        )

    def pddl_move_each_target_to_neighbor_location(self):
        t = self.scoped_targets()
        layer = self.binding["place_layer_name"]
        destinations = {o: self.pa[self.parent(o, layer)] for o in t}
        require(
            all(len(d) >= 2 for d in destinations.values())
            and len({tuple(sorted(d)) for d in destinations.values()}) >= 2,
            "neighbor destination choices are insufficient or identical",
        )
        return fold(
            "and",
            (
                fold(
                    "or",
                    (pddl_fact("object-in-place", o, p) for p in sorted_symbols(d)),
                )
                for o, d in sorted(destinations.items())
            ),
        )

    def pddl_inspect_targets_in_allowed_room_component(self):
        start = self.anchor_room()
        f = self.rooms("excluded")
        require(start not in f, "anchor room is forbidden")
        reached = set(self.paths("room", start, f))
        unrestricted = set(self.paths("room", start))
        t = self.objects()
        selected = {o for o in t if self.oroom[o] & reached}
        require(
            len(selected) >= 2
            and any(self.oroom[o] & (unrestricted - f - reached) for o in t - selected),
            "forbidden rooms disconnect no allowed targets",
        )
        return facts("safe", selected, "and")

    def legacy(self, family: str, name: str) -> str:
        """v1 semantics with the same typed assignment and selection pipeline."""
        b = self.binding
        o = b.get("object_symbol")
        other = b.get("other_object_symbol")
        r = b.get("room_symbol")
        r2 = b.get("other_room_symbol")
        p = b.get("place_symbol")
        p2 = b.get("other_place_symbol")
        t = self.objects()
        room_class = self.rooms("target")

        def nonempty(values, reason="empty result"):
            require(values, reason)
            return values

        if family == "qa":
            if name == "object_class_inventory":
                return class_answer(self.oc)
            if name == "object_total":
                return str(len(self.obj))
            if name == "room_total":
                return str(len(self.room))
            if name == "place_total":
                return str(len(self.place))
            if name == "room_class_inventory":
                return class_answer(self.rc)
            if name == "object_ids_for_class":
                require(len(t) >= 2, "class has only one member")
                return set_answer(t)
            if name == "object_count_for_class":
                require(len(t) >= 2, "class count is trivial")
                return str(len(t))
            if name == "object_centers_for_class":
                require(len(t) >= 2, "class has only one center")
                return sldp_set(
                    format_point(self.obj[x].position) for x in sorted_symbols(t)
                )
            if name == "object_class_lookup":
                label = self.obj[o].semantic_class or "unknown"
                require(
                    re.fullmatch(r"[A-Za-z_]\w*", label),
                    "class label is not a valid SLDP token",
                )
                return label
            if name == "object_center_lookup":
                return format_point(self.obj[o].position)
            if name == "rooms_for_class":
                return set_answer(nonempty(room_class))
            if name == "room_center_lookup":
                return format_point(self.room[r].position)
            if name == "parent_place_of_object":
                return set_answer(nonempty(self.op[o]))
            if name == "parent_room_of_object":
                return set_answer(nonempty(self.oroom[o]))
            if name == "parent_room_of_place":
                return set_answer(nonempty(self.place[p].parents & self.room.keys()))
            if name == "objects_in_room":
                return set_answer(nonempty(self.ro[r]))
            if name == "object_classes_in_room":
                return class_answer(
                    nonempty({self.obj[x].semantic_class for x in self.ro[r]} - {None})
                )
            if name == "object_count_in_room":
                require(self.ro[r], "empty room count")
                return str(len(self.ro[r]))
            if name == "class_objects_in_room":
                return set_answer(nonempty(t & self.ro[r]))
            if name == "class_count_in_room":
                return str(len(nonempty(t & self.ro[r])))
            if name == "places_in_room":
                return set_answer(nonempty(self.scene.places_by_room[r]))
            if name == "place_count_in_room":
                return str(len(nonempty(self.scene.places_by_room[r])))
            if name == "objects_in_place":
                return set_answer(nonempty(self.scene.objects_by_place[p]))
            if name == "object_classes_in_place":
                return class_answer(
                    nonempty(
                        {
                            self.obj[x].semantic_class
                            for x in self.scene.objects_by_place[p]
                        }
                        - {None}
                    )
                )
            if name == "object_count_in_place":
                return str(len(nonempty(self.scene.objects_by_place[p])))
            if name == "rooms_containing_class":
                return set_answer(nonempty(self.content_rooms(t)))
            if name == "room_count_containing_class":
                return str(len(nonempty(self.content_rooms(t))))
            if name == "room_neighbors":
                return set_answer(nonempty(self.ra[r]))
            if name == "room_degree":
                return str(len(self.ra[r]))
            if name in {"most_connected_rooms", "least_connected_rooms"}:
                return set_answer(
                    self.extrema(
                        {x: len(v) for x, v in self.ra.items()}, name.startswith("most")
                    )
                )
            if name in {
                "room_shortest_path",
                "room_hop_distance",
                "place_shortest_path",
            }:
                layer = "room" if name.startswith("room") else "place"
                a, z = (r, r2) if layer == "room" else (p, p2)
                path = self.scene.unique_shortest_paths(layer, a).get(z)
                require(path and len(path) > 2, "need a unique nontrivial route")
                return (
                    str(len(path) - 1)
                    if name == "room_hop_distance"
                    else sldp_list(path)
                )
            if name == "place_neighbors":
                return set_answer(nonempty(self.pa[p]))
            if name == "place_degree":
                return str(len(self.pa[p]))
            if name == "places_within_hops":
                paths = self.paths("place", p)
                h = b["hop_limit"]
                result = {x for x, path in paths.items() if 1 <= len(path) - 1 <= h}
                require(
                    result
                    and any(len(path) > 2 for x, path in paths.items() if x in result),
                    "hop budget is trivial",
                )
                return set_answer(result)
            if name == "objects_in_neighbor_places":
                return set_answer(
                    nonempty(
                        {x for a in self.pa[p] for x in self.scene.objects_by_place[a]}
                    )
                )
            if name in {
                "nearest_object",
                "farthest_object",
                "nearest_different_class_object",
            }:
                candidates = self.obj.keys() - {o}
                if name == "nearest_different_class_object":
                    candidates = {
                        x
                        for x in candidates
                        if self.obj[x].semantic_class != self.obj[o].semantic_class
                    }
                require(len(candidates) >= 2, "too few distance candidates")
                return set_answer(
                    self.extrema(
                        {x: self.distance(o, x) for x in candidates},
                        name == "farthest_object",
                    )
                )
            if name == "closest_cross_class_pair":
                return pair_answer(self.geometric_pairs())
            if name in {"nearest_place_to_object", "nearest_room_to_object"}:
                candidates = (
                    self.place if name == "nearest_place_to_object" else self.room
                )
                return set_answer(
                    self.extrema({x: self.distance(o, x) for x in candidates})
                )
            if name == "closest_room_pair":
                return pair_answer(
                    self.extrema(
                        {
                            (a, z): self.distance(a, z)
                            for a, z in combinations(self.room, 2)
                        }
                    )
                )
            if name in {"highest_object_of_class", "lowest_object_of_class"}:
                require(len(t) >= 2, "height class has only one member")
                return set_answer(
                    self.extrema(
                        {x: self.height(x) for x in t}, name.startswith("highest")
                    )
                )
            if name in {"nearest_object_to_room", "farthest_object_from_room"}:
                return set_answer(
                    self.extrema(
                        {x: self.distance(r, x) for x in self.obj},
                        name.startswith("farthest"),
                    )
                )
            if name == "objects_within_radius":
                result = {
                    x
                    for x in self.obj.keys() - {o}
                    if self.distance(o, x) <= b["radius_m"]
                }
                require(
                    result and len(result) < len(self.obj) - 1,
                    "radius does not divide candidates",
                )
                return set_answer(result)
        else:
            simple = {
                "visit_object_symbol": ("visited-object", o),
                "end_at_object_symbol": ("at-object", o),
                "inspect_object_symbol": ("safe", o),
                "hold_object_symbol": ("holding", o),
                "visit_room_symbol": ("visited-region", r),
                "end_in_room_symbol": ("in-region", r),
                "visit_place_symbol": ("visited-place", p),
                "end_at_place_symbol": ("at-place", p),
            }
            if name in simple:
                return pddl_fact(*simple[name])
            class_predicates = {
                "hold_object_class": "holding",
                "visit_object_class": "visited-object",
                "end_at_object_class": "at-object",
                "inspect_object_class": "safe",
            }
            if name in class_predicates:
                require(len(t) >= 2, "class choice is trivial")
                return facts(class_predicates[name], t)
            if name in {"visit_room_class", "end_in_room_class"}:
                require(len(room_class) >= 2, "room class choice is trivial")
                return facts(
                    "visited-region" if name.startswith("visit") else "in-region",
                    room_class,
                )
            if name in {
                "visit_room_containing_object",
                "end_in_room_containing_object",
            }:
                room = self.unique(self.oroom[o], "object has no unique room")
                return pddl_fact(
                    "visited-region" if name.startswith("visit") else "in-region", room
                )
            if name in {"visit_parent_place", "end_at_parent_place"}:
                return pddl_fact(
                    "visited-place" if name.startswith("visit") else "at-place",
                    self.parent(o),
                )
            distance_predicates = {
                "visit_nearest_object": ("visited-object", False),
                "inspect_nearest_object": ("safe", False),
                "hold_nearest_class_object": ("holding", False),
                "visit_farthest_object": ("visited-object", True),
                "end_at_farthest_object": ("at-object", True),
            }
            if name in distance_predicates:
                predicate, maximum = distance_predicates[name]
                candidates = (
                    t if name == "hold_nearest_class_object" else self.obj.keys()
                ) - {o}
                require(len(candidates) >= 2, "too few spatial candidates")
                return facts(
                    predicate,
                    self.extrema({x: self.distance(o, x) for x in candidates}, maximum),
                )
            if name in {"inspect_highest_class_object", "visit_lowest_class_object"}:
                require(len(t) >= 2, "height class has only one member")
                return facts(
                    "safe" if name.startswith("inspect") else "visited-object",
                    self.extrema(
                        {x: self.height(x) for x in t}, name.startswith("inspect")
                    ),
                )
            if name in {"visit_nearest_room", "end_in_nearest_room"}:
                return facts(
                    "visited-region" if name.startswith("visit") else "in-region",
                    self.extrema({x: self.distance(o, x) for x in self.room}),
                )
            if name in {"visit_most_connected_room", "end_in_least_connected_room"}:
                return facts(
                    "visited-region" if name.startswith("visit") else "in-region",
                    self.extrema(
                        {x: len(v) for x, v in self.ra.items()},
                        name.startswith("visit"),
                    ),
                )
            if name == "visit_adjacent_rooms":
                return facts("visited-region", nonempty(self.ra[r]), "and")
            if name == "visit_either_object":
                return facts("visited-object", [o, other])
            if name in {"visit_object_or_room", "visit_object_and_room"}:
                return fold(
                    "or" if "_or_" in name else "and",
                    [pddl_fact("visited-object", o), pddl_fact("visited-region", r)],
                )
            if name == "inspect_and_hold_object":
                return fold("and", [pddl_fact("safe", o), pddl_fact("holding", o)])
            ordered = {
                "visit_then_end_at_object": ("visited-object", o, "at-object", other),
                "visit_room_then_end_at_object": ("visited-region", r, "at-object", o),
                "visit_object_then_end_in_room": ("visited-object", o, "in-region", r),
                "visit_place_then_end_in_room": ("visited-place", p, "in-region", r),
                "visit_room_then_end_at_place": ("visited-region", r, "at-place", p),
                "visit_then_end_at_place": ("visited-place", p, "at-place", p2),
                "inspect_then_end_at_place": ("safe", o, "at-place", p),
                "hold_then_end_in_room": ("holding", o, "in-region", r),
            }
            if name in ordered:
                first, a, last, z = ordered[name]
                return fold("and", [pddl_fact(first, a), pddl_fact(last, z)])
            if name in {"move_object_to_place", "move_object_to_neighbor_place"}:
                require(p not in self.op[o], "relocation is already satisfied")
                if name == "move_object_to_neighbor_place":
                    require(
                        p in self.pa[self.parent(o)],
                        "destination is not adjacent to the current place",
                    )
                return pddl_fact("object-in-place", o, p)
            if name == "move_class_object_to_place":
                require(
                    len(t) >= 2 and all(p not in self.op[x] for x in t),
                    "need multiple nontrivial moves",
                )
                return fold(
                    "or",
                    (pddl_fact("object-in-place", x, p) for x in sorted_symbols(t)),
                )
            if name == "do_not_hold_object":
                return negated("holding", o)
            if name == "visit_without_holding":
                return fold(
                    "and", [pddl_fact("visited-object", o), negated("holding", o)]
                )
            if name == "visit_room_avoid_neighbor":
                require(r2 in self.ra[r], "avoided room is not a neighbor")
                return fold(
                    "and",
                    [pddl_fact("visited-region", r), negated("visited-region", r2)],
                )
            if name == "hold_class_or_inspect_object":
                require(len(t) >= 2, "class choice is trivial")
                return fold("or", [facts("holding", t), pddl_fact("safe", o)])
            if name == "visit_class_and_end_in_room":
                require(len(t) >= 2, "class choice is trivial")
                return fold(
                    "and", [facts("visited-object", t), pddl_fact("in-region", r)]
                )
        raise GenerationError(f"No grounding handler for {family}_{name}")


class Sampler:
    """Build finite eligible pools, then balance bindings across both tasks."""

    def __init__(
        self,
        scene: Scene,
        seed: int = 7,
        pool_size: int = 48,
        max_attempts: int = 30000,
    ):
        if pool_size < 1 or max_attempts < 1:
            raise GenerationError("pool_size and max_attempts must be positive")
        self.scene = scene
        self.seed = seed
        self.pool_size = pool_size
        self.max_attempts = max_attempts
        self.grounder = Grounder(scene)
        self.class_use = Counter()
        self.entity_use = Counter()
        self.pair_use = Counter()

    def rng(self, identifier):
        digest = hashlib.sha256(f"{self.seed}:{identifier}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def missing_capabilities(self, entry):
        g = self.grounder
        s = self.scene
        available = {
            "object_classes": bool(g.oc),
            "room_classes": bool(g.rc),
            "containment": any(g.op.values()) or any(p.parents for p in s.places),
            "navigable_places": bool(g.place),
            "selected_place_layer": bool(g.layers),
            "object_centers": bool(g.obj)
            and all(all(math.isfinite(x) for x in o.position) for o in s.objects),
            "room_centers": bool(g.room)
            and all(all(math.isfinite(x) for x in r.position) for r in s.rooms),
            "place_centers": bool(g.place)
            and all(all(math.isfinite(x) for x in p.position) for p in s.places),
            # Spark/Hydra's documented default is z-up; custom scenes override it.
            "vertical_axis": s._metadata.get("question_generation", {}).get(
                "up_axis", "z"
            )
            in {"x", "y", "z"},
            "room_connectivity": any(g.ra.values()),
            "place_connectivity": any(g.pa.values()),
            "selected_place_connectivity": any(g.pa.values()),
        }
        return [c for c in entry["requires"] if not available[c]]

    def base_bindings(self, entry, rng):
        roles = {parameter_role(n, t): t for n, t in entry["parameters"].items()}
        domains = {}
        for role, typ in roles.items():
            if typ.startswith("object_class"):
                domains[role] = list(self.grounder.oc)
            elif typ.startswith("room_class"):
                domains[role] = list(self.grounder.rc)
            elif typ == "object_symbol":
                domains[role] = list(self.grounder.obj)
            elif typ == "room_symbol":
                domains[role] = list(self.grounder.room)
            elif typ == "place_symbol":
                domains[role] = list(self.grounder.place)
            elif typ == "place_layer_name":
                domains[role] = sorted(self.grounder.layers)
        # Join scoped references to their room category before traversing the
        # product. Invalid references never consume numerical evaluation attempts.
        units = {}
        scopes = REFERENCE_SCOPES.get(entry["id"], {})
        scoped = {}
        for role, room_role in scopes.items():
            key = role + "_object_class"
            if room_role is None:
                domains[key] = [
                    c for c in domains[key] if len(self.grounder.oc[c]) == 1
                ]
            else:
                scoped.setdefault(room_role + "_room_class", []).append(key)
        for room_key, reference_keys in sorted(scoped.items()):
            reference_keys = sorted(reference_keys)
            joint = []
            for room_class in domains.pop(room_key):
                contents = set().union(
                    *(self.grounder.ro[r] for r in self.grounder.rc[room_class])
                )
                choices = [
                    [c for c in domains[k] if len(self.grounder.oc[c] & contents) == 1]
                    for k in reference_keys
                ]
                for values in product(*choices):
                    joint.append(
                        dict(zip([room_key, *reference_keys], [room_class, *values]))
                    )
            for key in reference_keys:
                del domains[key]
            units[room_key] = joint
        units.update(
            {key: [{key: v} for v in values] for key, values in domains.items()}
        )
        keys = sorted(units)
        # Shuffle the Cartesian index space without constructing a potentially
        # enormous product for larger scenes. A coprime stride visits every index.
        size = math.prod(len(units[k]) for k in keys)
        if not size:
            return
        offset = rng.randrange(size)
        stride = rng.randrange(1, size + 1)
        while math.gcd(stride, size) != 1:
            stride = rng.randrange(1, size + 1)
        for step in range(size):
            index = (offset + step * stride) % size
            values = {}
            for key in reversed(keys):
                index, digit = divmod(index, len(units[key]))
                values.update(units[key][digit])
            # Independent grammatical forms of a role already share one key.
            groups = {}
            for key, value in values.items():
                typ = roles[key]
                group = typ.split("_class")[0] + "_class" if "_class" in typ else typ
                if group in {
                    "object_class",
                    "room_class",
                    "object_symbol",
                    "room_symbol",
                    "place_symbol",
                }:
                    groups.setdefault(group, []).append(value)
            if all(len(v) == len(set(v)) for v in groups.values()):
                yield values

    @staticmethod
    def spread(values, limit=24):
        """Retain endpoints and evenly spaced observations, without random caps."""
        ordered = sorted(set(values))
        if len(ordered) <= limit:
            return ordered
        return [
            ordered[round(i * (len(ordered) - 1) / (limit - 1))] for i in range(limit)
        ]

    def radius_candidates(self, base):
        g = self.grounder
        targets = g.objects() or set(g.obj)
        references = [
            g.objects(role) for role in ("reference", "excluded", "comparison")
        ]
        if base.get("object_symbol"):
            references.append({base["object_symbol"]})
        references = [v for v in references if v] or [set(g.obj)]
        observed = []
        local = []
        for refs in references:
            for target in sorted_symbols(targets):
                values = [
                    g.distance(target, ref) for ref in sorted_symbols(refs - {target})
                ]
                if values:
                    observed.extend((min(values), max(values)))
            for room in sorted_symbols(g.rooms()):
                for target in sorted_symbols(targets & g.ro[room]):
                    values = [
                        g.distance(target, ref)
                        for ref in sorted_symbols(refs & g.ro[room] - {target})
                    ]
                    if values:
                        local.extend((min(values), max(values)))

        def thresholds(values, limit):
            ordered = self.spread(values, limit)
            middle = [round((a + b) / 2, 2) for a, b in pairwise(ordered)]
            # Also sample boundaries at display precision. Grounding always uses
            # exactly the displayed value, including <= and > boundary semantics.
            return [*middle, *(round(v, 2) for v in ordered)]

        radii = {1.0, 2.0, 3.0, 5.0, 10.0}
        radii.update(thresholds(observed, 16))
        radii.update(thresholds(local, 16))
        return sorted(r for r in radii if r > 0)

    def hop_candidates(self, entry, base):
        g = self.grounder
        if "place_symbol" in base:
            # v1 explicitly asks for a small neighborhood around a supplied ID.
            distances = g.hop_distances("place", base["place_symbol"])
            return list(range(2, min(6, max(distances.values(), default=0)) + 1))
        scopes = REFERENCE_SCOPES.get(entry["id"], {})
        ref = g.reference("reference", scopes.get("reference"))
        if "place_layer_name" in base:
            start = g.parent(ref, base["place_layer_name"])
            distances = g.hop_distances("place", start)
            if entry["id"] == "qa_common_place_hop_neighborhood":
                other = g.parent(g.reference("comparison"), base["place_layer_name"])
                comparison = g.hop_distances("place", other)
                observed = [
                    max(h, comparison[p])
                    for p, h in distances.items()
                    if p in comparison
                ]
            elif g.objects():
                observed = [
                    min(distances[p] for p in g.op[o] if p in distances)
                    for o in sorted_symbols(g.objects())
                    if g.op[o] & distances.keys()
                ]
            else:
                observed = list(distances.values())
        else:
            start = g.unique(g.oroom[ref], "reference has no unique containing room")
            distances = g.hop_distances("room", start)
            observed = list(distances.values())
        values = {2, 3, 4, 5, 6}
        for value in self.spread(observed):
            values.update((value - 1, value))
        return sorted(v for v in values if v >= 2)

    def numeric_bindings(self, entry, base, rng):
        numeric = {
            n: t
            for n, t in entry["parameters"].items()
            if t in {"distance_m", "positive_integer"}
        }
        if not numeric:
            yield base
            return
        g = self.grounder
        g.binding = base
        radii = self.radius_candidates(base) if "distance_m" in numeric.values() else []
        domains = {}
        for name, typ in numeric.items():
            if typ == "distance_m":
                domains[name] = radii
            elif name == "required_count":
                domains[name] = sorted({len(t) for t in g.room_groups().values() if t})
            elif name == "top_k":
                groups = g.eligible_groups()
                counts = [len(t) for t, _ in groups.values()]
                domains[name] = [
                    k
                    for k in self.spread(range(2, min(counts, default=1) + 1), 8)
                    if max(counts, default=0) > k
                ]
            elif name == "hop_limit":
                domains[name] = self.hop_candidates(entry, base)
            else:
                domains[name] = [2, 3, 4, 5, 6]
        keys = sorted(domains)
        # Walk the index space without constructing quadratic distance-band
        # products. A random offset and coprime stride retain reproducibility.
        size = math.prod(len(domains[k]) for k in keys)
        if not size:
            return
        offset = rng.randrange(size)
        stride = rng.randrange(1, size + 1)
        while math.gcd(stride, size) != 1:
            stride = rng.randrange(1, size + 1)
        for step in range(size):
            index = (offset + step * stride) % size
            binding = dict(base)
            for key in reversed(keys):
                index, digit = divmod(index, len(domains[key]))
                binding[key] = domains[key][digit]
            yield binding

    def candidates(self, entry):
        missing = self.missing_capabilities(entry)
        if missing:
            return [], {
                "status": "unsupported",
                "reason": "missing capabilities: " + ", ".join(missing),
                "attempts": 0,
            }
        rng = self.rng(entry["id"])
        pool = []
        rejected = Counter()
        attempts = 0
        base_assignments = 0
        preparation_rejections = Counter()
        seen = set()
        complete = True
        for base in self.base_bindings(entry, rng):
            if base_assignments >= self.max_attempts:
                complete = False
                break
            base_assignments += 1
            produced = False
            best = None
            try:
                for binding in self.numeric_bindings(entry, base, rng):
                    produced = True
                    if attempts >= self.max_attempts:
                        complete = False
                        break
                    attempts += 1
                    try:
                        candidate = self.grounder.evaluate(entry, binding)
                    except Ineligible as exc:
                        rejected[str(exc)] += 1
                        continue
                    penalty = sum(
                        not passed for passed in candidate.preferences.values()
                    )
                    if best is None or penalty < sum(
                        not passed for passed in best.preferences.values()
                    ):
                        best = candidate
                    if penalty == 0:
                        break
                if not produced:
                    preparation_rejections["no scene-derived numeric values"] += 1
            except Ineligible as exc:
                preparation_rejections[str(exc)] += 1
            if best is not None:
                fingerprint = (
                    tuple(sorted(best.question.parameters.items())),
                    best.question.solution,
                )
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    pool.append(best)
            if len(pool) >= self.pool_size:
                complete = False
                break
            if attempts >= self.max_attempts:
                complete = False
                break
        status = (
            "generated" if pool else ("unsupported" if complete else "search_exhausted")
        )
        return pool, {
            "status": status,
            "attempts": attempts,
            "base_assignments": base_assignments,
            "preparation_rejections": dict(preparation_rejections.most_common(5)),
            "eligible_candidates": len(pool),
            "search_complete": complete,
            "rejections": dict(rejected.most_common(5)),
            **(
                {
                    "reason": "no assignment satisfies the implemented eligibility checks"
                    if complete
                    else "candidate search budget exhausted"
                }
                if not pool
                else {}
            ),
        }

    def select(self, catalog):
        pools = {}
        report = {}
        entries = [e for family in ("qa", "pddl") for e in catalog[family]]
        for entry in entries:
            pool, details = self.candidates(entry)
            pools[entry["id"]] = pool
            report[entry["id"]] = details
        selected = {}

        # Scarce families choose first; balance the flexible families around them.
        def order(entry):
            return len(pools[entry["id"]]), hashlib.sha256(
                f"{self.seed}:{entry['id']}".encode()
            ).hexdigest()

        for entry in sorted(entries, key=order):
            pool = pools[entry["id"]]
            if not pool:
                continue

            def features(candidate):
                classes = [
                    (role, value)
                    for role, value in candidate.bindings.items()
                    if role.endswith("_class")
                ]
                entities = candidate.references | candidate.answer_entities
                pairs = list(combinations(sorted(value for _, value in classes), 2))
                return classes, entities, pairs

            def score(candidate):
                classes, entities, pairs = features(candidate)
                # Normalize entity use so a large set answer cannot dominate cost.
                entity_cost = sum(self.entity_use[x] for x in entities) / max(
                    1, len(entities)
                )
                return (
                    sum(not passed for passed in candidate.preferences.values()),
                    sum(self.class_use[x] for x in classes)
                    + entity_cost
                    + sum(self.pair_use[x] for x in pairs),
                )

            candidate = min(pool, key=score)
            selected[entry["id"]] = candidate.question
            classes, entities, pairs = features(candidate)
            self.class_use.update(classes)
            self.entity_use.update(entities)
            self.pair_use.update(pairs)
            report[entry["id"]]["bindings"] = candidate.bindings
            report[entry["id"]]["difficulty_preferences"] = candidate.preferences
            report[entry["id"]]["resolved_references"] = sorted_symbols(
                candidate.references
            )
        self.report = {
            "families": report,
            "diversity": {
                "class_role_counts": {
                    role: {
                        v: n
                        for (r, v), n in sorted(self.class_use.items())
                        if r == role
                    }
                    for role in sorted({r for r, _ in self.class_use})
                },
                "entity_counts": dict(sorted(self.entity_use.items())),
                "class_pair_counts": {
                    " / ".join(pair): n for pair, n in sorted(self.pair_use.items())
                },
            },
            "settings": {
                "seed": self.seed,
                "pool_size": self.pool_size,
                "max_attempts": self.max_attempts,
                "max_base_assignments": self.max_attempts,
                "selection_policy": "difficulty_preferences_then_diversity",
            },
        }
        return {
            family: {
                e["id"]: selected[e["id"]]
                for e in catalog[family]
                if e["id"] in selected
            }
            for family in ("qa", "pddl")
        }

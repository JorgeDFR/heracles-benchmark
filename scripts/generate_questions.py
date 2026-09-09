#!/usr/bin/env python3
"""Generate deterministic QA and PDDL benchmark questions from a Spark DSG.

The generator creates one grounded question for each type declared in
``data/questions/question_types.yaml``: 50 scene-graph QA questions and 50 PDDL
goals. Entity choices and wording are reproducible for a given random seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import random
import re
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import spark_dsg
except ModuleNotFoundError:  # Report a concise, actionable error from main().
    spark_dsg = None

try:
    import yaml
except ModuleNotFoundError:  # Report a concise, actionable error from main().
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "data" / "scene_graphs" / "example_dsg.json"
DEFAULT_CATALOG = ROOT / "data" / "questions" / "question_types.yaml"
DEFAULT_QUESTION_ROOT = ROOT / "data" / "questions"
GENERATOR_VERSION = 1
LOCAL_ID_MASK = (1 << 56) - 1

LAYER_NAMES = {
    "object": "OBJECTS",
    "place": "PLACES",
    "mesh_place": "MESH_PLACES",
    "room": "ROOMS",
}

# Common Hydra/Spark DSG labelspace names used in graph metadata.
LABELSPACE_KEYS = {
    "object": ("_l2p0", "objects", "object_labelspace"),
    "mesh_place": ("_l3p1", "mesh", "mesh_places"),
    "room": ("room_labelspace", "_l4p0", "rooms"),
}

SYNONYMS = {
    "trash": ("waste bin", "trash can", "rubbish bin"),
    "storage": ("storage unit", "cabinet", "storage fixture"),
    "seating": ("seat", "piece of seating", "seating object"),
    "decor": ("decoration", "decorative item", "ornament"),
    "appliance": ("appliance", "electrical appliance", "device"),
    "light": ("light fixture", "lamp", "light"),
    "sign": ("signage", "sign", "notice"),
    "box": ("container", "box", "carton"),
    "bicycle": ("bike", "bicycle", "cycle"),
    "hallway": ("corridor", "hallway", "passage"),
    "lounge": ("sitting room", "lounge", "common room"),
}


class GenerationError(RuntimeError):
    """Raised when a graph cannot support one of the declared question types."""


class NoAliasDumper(yaml.SafeDumper if yaml else object):
    """Keep repeated YAML values readable instead of emitting anchors."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass(frozen=True)
class Entity:
    symbol: str
    kind: str
    semantic_class: str | None
    position: tuple[float, float, float]
    parents: frozenset[str]
    neighbors: frozenset[str]
    children: frozenset[str]

    @property
    def parent(self) -> str | None:
        """Return a stable parent when a single-parent operation requires one."""
        ordered = sorted_symbols(self.parents)
        return ordered[0] if ordered else None


@dataclass(frozen=True)
class Question:
    question: str
    solution: str


def symbol_key(symbol: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([^0-9]*)([0-9]+)", symbol)
    if not match:
        return (symbol, -1, symbol)
    return (match.group(1), int(match.group(2)), symbol)


def sorted_symbols(values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=symbol_key)


def relation_values(node: Any, attribute: str) -> list[Any]:
    value = getattr(node, attribute, ())
    value = value() if callable(value) else value
    return list(value or ())


def node_symbol(node_id: Any) -> str:
    if hasattr(node_id, "str"):
        return node_id.str(True)
    raw_id = int(node_id)
    return f"{chr(raw_id >> 56)}{raw_id & LOCAL_ID_MASK}"


def point(position: Any) -> tuple[float, float, float]:
    return (float(position[0]), float(position[1]), float(position[2]))


def format_number(value: float, digits: int = 2) -> str:
    rounded = round(value, digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def format_point(position: Sequence[float]) -> str:
    return "POINT(" + " ".join(format_number(value) for value in position) + ")"


def indefinite(noun_phrase: str) -> str:
    article = "an" if noun_phrase[:1].casefold() in "aeiou" else "a"
    return f"{article} {noun_phrase}"


def sldp_set(values: Iterable[str]) -> str:
    return "<" + ", ".join(values) + ">"


def sldp_list(values: Iterable[str]) -> str:
    return "[" + ", ".join(values) + "]"


def pddl_fact(predicate: str, symbol: str, destination: str | None = None) -> str:
    suffix = f" {destination}" if destination is not None else ""
    return f"({predicate} {symbol}{suffix})"


def pddl_group(operator: str, clauses: Iterable[str], *, always: bool = False) -> str:
    materialized = list(clauses)
    if not materialized:
        raise GenerationError(f"Cannot construct an empty PDDL `{operator}` goal")
    if len(materialized) == 1 and not always:
        return materialized[0]
    return f"({operator} {' '.join(materialized)})"


def euclidean(left: Entity, right: Entity) -> float:
    return math.dist(left.position, right.position)


class Scene:
    """Small, version-tolerant index over the Spark DSG Python API."""

    def __init__(self, graph: Any):
        self.graph = graph
        self._metadata = graph.metadata.get() or {}
        self.entities: dict[str, Entity] = {}
        self.objects = self._read_layer("object")
        places_3d = self._read_layer("place")
        places_2d = self._read_layer("mesh_place")
        self.places = sorted(
            places_3d + places_2d, key=lambda item: symbol_key(item.symbol)
        )
        self.rooms = self._read_layer("room")
        self.entities = {
            entity.symbol: entity for entity in self.objects + self.places + self.rooms
        }

        self.objects_by_class = self._group_by_class(self.objects)
        self.rooms_by_class = self._group_by_class(self.rooms)
        self.objects_by_place = {
            place.symbol: sorted_symbols(
                obj.symbol for obj in self.objects if place.symbol in obj.parents
            )
            for place in self.places
        }
        self.places_by_room = {
            room.symbol: sorted_symbols(
                place.symbol for place in self.places if room.symbol in place.parents
            )
            for room in self.rooms
        }
        self.objects_by_room = {
            room.symbol: sorted_symbols(
                obj.symbol
                for obj in self.objects
                if room.symbol in self.rooms_for_object(obj.symbol)
            )
            for room in self.rooms
        }

    @classmethod
    def load(cls, path: Path) -> Scene:
        return cls(spark_dsg.DynamicSceneGraph.load(str(path)))

    def _nodes(self, kind: str) -> list[Any]:
        layer = getattr(spark_dsg.DsgLayers, LAYER_NAMES[kind])
        try:
            return list(self.graph.get_layer(layer).nodes)
        except (IndexError, KeyError, RuntimeError):
            return []

    def _labelspace(self, kind: str) -> dict[int, str]:
        labelspaces = self._metadata.get("labelspaces", {})
        for key in LABELSPACE_KEYS.get(kind, ()):
            if key in labelspaces:
                return {int(label): str(name) for label, name in labelspaces[key]}
        return {}

    def _semantic_class(self, node: Any, kind: str) -> str | None:
        semantic_label = getattr(node.attributes, "semantic_label", None)
        if semantic_label is None:
            return None
        try:
            label = int(semantic_label)
        except (TypeError, ValueError):
            return None
        known = self._labelspace(kind)
        value = known.get(label)
        if value and value.lower() not in {"unknown", "none"}:
            return value
        return None

    def _read_layer(self, kind: str) -> list[Entity]:
        result = []
        for node in self._nodes(kind):
            parents = relation_values(node, "parents")
            if not parents and hasattr(node, "has_parent") and node.has_parent():
                parents = [node.get_parent()]
            result.append(
                Entity(
                    symbol=node_symbol(node.id),
                    kind=kind,
                    semantic_class=self._semantic_class(node, kind),
                    position=point(node.attributes.position),
                    parents=frozenset(node_symbol(value) for value in parents),
                    neighbors=frozenset(
                        node_symbol(value)
                        for value in relation_values(node, "siblings")
                    ),
                    children=frozenset(
                        node_symbol(value)
                        for value in relation_values(node, "children")
                    ),
                )
            )
        return sorted(result, key=lambda item: symbol_key(item.symbol))

    @staticmethod
    def _group_by_class(entities: Iterable[Entity]) -> dict[str, list[Entity]]:
        result: dict[str, list[Entity]] = {}
        for entity in entities:
            if entity.semantic_class:
                result.setdefault(entity.semantic_class, []).append(entity)
        return dict(sorted(result.items()))

    def entity(self, symbol: str) -> Entity:
        return self.entities[symbol]

    def rooms_for_object(self, object_symbol: str) -> list[str]:
        obj = next(
            (item for item in self.objects if item.symbol == object_symbol), None
        )
        if not obj:
            return []
        places = {item.symbol: item for item in self.places}
        return sorted_symbols(
            room_symbol
            for place_symbol in obj.parents
            if place_symbol in places
            for room_symbol in places[place_symbol].parents
        )

    def adjacency(self, kind: str) -> dict[str, set[str]]:
        entities = self.rooms if kind == "room" else self.places
        valid = {entity.symbol for entity in entities}
        return {
            entity.symbol: set(entity.neighbors).intersection(valid)
            for entity in entities
        }

    def unique_shortest_paths(self, kind: str, start: str) -> dict[str, list[str]]:
        """Find all destinations having a unique shortest path from ``start``."""
        adjacency = self.adjacency(kind)
        queue: deque[str] = deque([start])
        distance = {start: 0}
        path_count = {start: 1}
        predecessor: dict[str, str] = {}
        while queue:
            current = queue.popleft()
            for neighbor in sorted_symbols(adjacency.get(current, set())):
                proposed_distance = distance[current] + 1
                if neighbor not in distance:
                    distance[neighbor] = proposed_distance
                    path_count[neighbor] = path_count[current]
                    predecessor[neighbor] = current
                    queue.append(neighbor)
                elif distance[neighbor] == proposed_distance:
                    path_count[neighbor] = min(
                        2, path_count[neighbor] + path_count[current]
                    )
        paths: dict[str, list[str]] = {}
        for goal, count in path_count.items():
            if count != 1:
                continue
            path = [goal]
            while path[-1] != start:
                path.append(predecessor[path[-1]])
            paths[goal] = list(reversed(path))
        return paths


class Choices:
    def __init__(self, scene: Scene, seed: int):
        self.scene = scene
        self.rng = random.Random(seed)

    def one(self, values: Sequence[Any], reason: str) -> Any:
        if not values:
            raise GenerationError(f"Scene has no candidate for {reason}")
        return self.rng.choice(list(values))

    def two(self, values: Sequence[Any], reason: str) -> tuple[Any, Any]:
        if len(values) < 2:
            raise GenerationError(f"Scene needs at least two candidates for {reason}")
        first, second = self.rng.sample(list(values), 2)
        return first, second

    def class_name(self, *, multiple: bool = False) -> str:
        classes = [
            name
            for name, entities in self.scene.objects_by_class.items()
            if not multiple or len(entities) > 1
        ]
        return self.one(classes, "an object semantic class")

    def room_class(self, *, multiple: bool = False) -> str:
        classes = [
            name
            for name, entities in self.scene.rooms_by_class.items()
            if not multiple or len(entities) > 1
        ]
        return self.one(classes, "a room semantic class")

    def vocabulary(self, semantic_class: str) -> str:
        return self.rng.choice(SYNONYMS.get(semantic_class.lower(), (semantic_class,)))


def extrema(
    reference: Entity,
    candidates: Iterable[Entity],
    *,
    farthest: bool = False,
) -> list[Entity]:
    scored = [(euclidean(reference, candidate), candidate) for candidate in candidates]
    if not scored:
        raise GenerationError("A distance question has no comparison candidates")
    target = (max if farthest else min)(score for score, _ in scored)
    return [
        candidate
        for score, candidate in scored
        if math.isclose(score, target, abs_tol=1e-9)
    ]


def axis_extrema(
    candidates: Sequence[Entity], axis: int, maximum: bool
) -> list[Entity]:
    if not candidates:
        raise GenerationError("An extrema question has no candidates")
    target = (max if maximum else min)(entity.position[axis] for entity in candidates)
    return [
        entity
        for entity in candidates
        if math.isclose(entity.position[axis], target, abs_tol=1e-9)
    ]


def pddl_for_entities(
    predicate: str,
    entities: Iterable[Entity],
    *,
    operator: str = "or",
    force_operator: bool = False,
) -> str:
    clauses = [pddl_fact(predicate, entity.symbol) for entity in entities]
    return pddl_group(operator, clauses, always=force_operator)


def pair_with_path(scene: Scene, kind: str) -> tuple[Entity, Entity, list[str]]:
    entities = scene.rooms if kind == "room" else scene.places
    candidates = []
    for index, left in enumerate(entities):
        paths = scene.unique_shortest_paths(kind, left.symbol)
        for right in entities[index + 1 :]:
            path = paths.get(right.symbol)
            if path and len(path) > 1:
                candidates.append((left, right, path))
    if not candidates:
        raise GenerationError(f"Scene has no connected pair of {kind} nodes")
    # Favor a nontrivial graph route and use symbols for deterministic tie-breaking.
    return max(candidates, key=lambda item: (len(item[2]), symbol_key(item[0].symbol)))


def qa_questions(scene: Scene, choices: Choices) -> dict[str, Question]:
    q: dict[str, Question] = {}
    obj = choices.one(scene.objects, "an object")
    room = choices.one(scene.rooms, "a room")
    class_name = choices.class_name()
    class_word = choices.vocabulary(class_name)
    room_class = choices.room_class()

    q["qa_object_class_inventory"] = Question(
        "Give the complete set of object categories represented in this map.",
        sldp_set(sorted(scene.objects_by_class)),
    )
    q["qa_object_total"] = Question(
        "How many object nodes does the scene contain?", str(len(scene.objects))
    )
    q["qa_room_total"] = Question(
        "What is the total number of mapped rooms?", str(len(scene.rooms))
    )
    q["qa_place_total"] = Question(
        "Count all navigable place nodes, combining the 2D and 3D place layers.",
        str(len(scene.places)),
    )
    q["qa_room_class_inventory"] = Question(
        "Which distinct room categories occur in the scene?",
        sldp_set(sorted(scene.rooms_by_class)),
    )
    q["qa_object_ids_for_class"] = Question(
        f"Return the IDs of every object categorized as {class_word} in the map.",
        sldp_set(entity.symbol for entity in scene.objects_by_class[class_name]),
    )
    count_class = choices.class_name()
    q["qa_object_count_for_class"] = Question(
        f"How many {choices.vocabulary(count_class)} objects are represented?",
        str(len(scene.objects_by_class[count_class])),
    )
    center_class = choices.class_name()
    q["qa_object_centers_for_class"] = Question(
        f"Report all 3D center points for the {choices.vocabulary(center_class)} objects.",
        sldp_set(
            format_point(entity.position)
            for entity in scene.objects_by_class[center_class]
        ),
    )
    q["qa_object_class_lookup"] = Question(
        f"What semantic category is assigned to {obj.symbol}?",
        obj.semantic_class or "unknown",
    )
    center_obj = choices.one(scene.objects, "an object center lookup")
    q["qa_object_center_lookup"] = Question(
        f"Where is the center of object {center_obj.symbol}?",
        format_point(center_obj.position),
    )
    q["qa_rooms_for_class"] = Question(
        f"List the room IDs categorized as {choices.vocabulary(room_class)}.",
        sldp_set(entity.symbol for entity in scene.rooms_by_class[room_class]),
    )
    q["qa_room_center_lookup"] = Question(
        f"Give the center point recorded for room {room.symbol}.",
        format_point(room.position),
    )

    objects_with_place = [entity for entity in scene.objects if entity.parents]
    child_obj = choices.one(objects_with_place, "an object with a parent place")
    q["qa_parent_place_of_object"] = Question(
        f"Which place nodes directly contain {child_obj.symbol}?",
        sldp_set(sorted_symbols(child_obj.parents)),
    )
    objects_with_room = [
        entity for entity in scene.objects if scene.rooms_for_object(entity.symbol)
    ]
    room_obj = choices.one(objects_with_room, "an object contained in a room")
    q["qa_parent_room_of_object"] = Question(
        f"Following containment through places, which rooms contain {room_obj.symbol}?",
        sldp_set(scene.rooms_for_object(room_obj.symbol)),
    )
    places_with_room = [entity for entity in scene.places if entity.parents]
    room_place = choices.one(places_with_room, "a place contained in a room")
    q["qa_parent_room_of_place"] = Question(
        f"Name every room that is a parent of place {room_place.symbol}.",
        sldp_set(sorted_symbols(room_place.parents)),
    )

    rooms_with_objects = [r for r in scene.rooms if scene.objects_by_room[r.symbol]]
    populated_room = choices.one(rooms_with_objects, "a room containing objects")
    populated_ids = scene.objects_by_room[populated_room.symbol]
    q["qa_objects_in_room"] = Question(
        f"Which object IDs descend from room {populated_room.symbol}?",
        sldp_set(populated_ids),
    )
    class_room = choices.one(rooms_with_objects, "a room containing object classes")
    classes_in_room = sorted(
        {
            scene.entity(symbol).semantic_class
            for symbol in scene.objects_by_room[class_room.symbol]
        }
        - {None}
    )
    q["qa_object_classes_in_room"] = Question(
        f"What kinds of objects can be found inside {class_room.symbol}?",
        sldp_set(classes_in_room),
    )
    count_room = choices.one(scene.rooms, "a room object count")
    q["qa_object_count_in_room"] = Question(
        f"Count the objects contained under room {count_room.symbol}.",
        str(len(scene.objects_by_room[count_room.symbol])),
    )
    room_class_pairs = [
        (r, class_value)
        for r in rooms_with_objects
        for class_value in sorted(
            {
                scene.entity(symbol).semantic_class
                for symbol in scene.objects_by_room[r.symbol]
            }
            - {None}
        )
    ]
    filtered_room, filtered_class = choices.one(room_class_pairs, "a room/class pair")
    filtered_ids = [
        symbol
        for symbol in scene.objects_by_room[filtered_room.symbol]
        if scene.entity(symbol).semantic_class == filtered_class
    ]
    q["qa_class_objects_in_room"] = Question(
        f"Identify each {choices.vocabulary(filtered_class)} located in {filtered_room.symbol}.",
        sldp_set(filtered_ids),
    )
    count_pair = choices.one(room_class_pairs, "a room/class count pair")
    count_ids = [
        symbol
        for symbol in scene.objects_by_room[count_pair[0].symbol]
        if scene.entity(symbol).semantic_class == count_pair[1]
    ]
    q["qa_class_count_in_room"] = Question(
        f"How many {choices.vocabulary(count_pair[1])} objects belong to {count_pair[0].symbol}?",
        str(len(count_ids)),
    )
    q["qa_places_in_room"] = Question(
        f"Enumerate the place nodes immediately below {room.symbol}.",
        sldp_set(scene.places_by_room[room.symbol]),
    )
    place_count_room = choices.one(scene.rooms, "a room place count")
    q["qa_place_count_in_room"] = Question(
        f"How many direct place children does {place_count_room.symbol} have?",
        str(len(scene.places_by_room[place_count_room.symbol])),
    )

    populated_places = [p for p in scene.places if scene.objects_by_place[p.symbol]]
    populated_place = choices.one(populated_places, "a place containing objects")
    q["qa_objects_in_place"] = Question(
        f"List the objects attached directly to place {populated_place.symbol}.",
        sldp_set(scene.objects_by_place[populated_place.symbol]),
    )
    class_place = choices.one(populated_places, "a place containing object classes")
    q["qa_object_classes_in_place"] = Question(
        f"Which object categories are directly contained by {class_place.symbol}?",
        sldp_set(
            sorted(
                {
                    scene.entity(symbol).semantic_class
                    for symbol in scene.objects_by_place[class_place.symbol]
                }
                - {None}
            )
        ),
    )
    count_place = choices.one(scene.places, "a place object count")
    q["qa_object_count_in_place"] = Question(
        f"How many object children are assigned to {count_place.symbol}?",
        str(len(scene.objects_by_place[count_place.symbol])),
    )
    classes_in_rooms = [
        name
        for name in scene.objects_by_class
        if any(
            scene.rooms_for_object(entity.symbol)
            for entity in scene.objects_by_class[name]
        )
    ]
    containing_class = choices.one(
        classes_in_rooms, "an object class assigned to rooms"
    )
    containing_rooms = sorted_symbols(
        room_symbol
        for entity in scene.objects_by_class[containing_class]
        for room_symbol in scene.rooms_for_object(entity.symbol)
    )
    q["qa_rooms_containing_class"] = Question(
        f"Which rooms contain at least one {choices.vocabulary(containing_class)}?",
        sldp_set(containing_rooms),
    )
    room_count_class = choices.one(
        classes_in_rooms, "an object class for room counting"
    )
    room_count_symbols = {
        room_symbol
        for entity in scene.objects_by_class[room_count_class]
        for room_symbol in scene.rooms_for_object(entity.symbol)
    }
    q["qa_room_count_containing_class"] = Question(
        f"In how many rooms does a {choices.vocabulary(room_count_class)} appear?",
        str(len(room_count_symbols)),
    )

    room_adjacency = scene.adjacency("room")
    connected_rooms = [r for r in scene.rooms if room_adjacency[r.symbol]]
    neighbor_room = choices.one(connected_rooms, "a room with a neighbor")
    q["qa_room_neighbors"] = Question(
        f"Which rooms share a direct connectivity edge with {neighbor_room.symbol}?",
        sldp_set(sorted_symbols(room_adjacency[neighbor_room.symbol])),
    )
    degree_room = choices.one(scene.rooms, "a room degree")
    q["qa_room_degree"] = Question(
        f"What is the connectivity degree of room {degree_room.symbol}?",
        str(len(room_adjacency[degree_room.symbol])),
    )
    max_degree = max(map(len, room_adjacency.values()))
    min_degree = min(map(len, room_adjacency.values()))
    q["qa_most_connected_rooms"] = Question(
        "Return every room tied for the greatest number of direct room connections.",
        sldp_set(
            sorted_symbols(
                symbol
                for symbol, links in room_adjacency.items()
                if len(links) == max_degree
            )
        ),
    )
    q["qa_least_connected_rooms"] = Question(
        "Which rooms are tied at the minimum room-connectivity degree?",
        sldp_set(
            sorted_symbols(
                symbol
                for symbol, links in room_adjacency.items()
                if len(links) == min_degree
            )
        ),
    )
    path_left, path_right, room_path = pair_with_path(scene, "room")
    q["qa_room_shortest_path"] = Question(
        f"Give one shortest room-by-room route from {path_left.symbol} to {path_right.symbol}, including both ends.",
        sldp_list(room_path),
    )
    q["qa_room_hop_distance"] = Question(
        f"How many room edges separate {path_left.symbol} from {path_right.symbol} along a shortest route?",
        str(len(room_path) - 1),
    )

    place_adjacency = scene.adjacency("place")
    connected_places = [p for p in scene.places if place_adjacency[p.symbol]]
    neighbor_place = choices.one(connected_places, "a connected place")
    q["qa_place_neighbors"] = Question(
        f"Return the immediate navigation neighbors of place {neighbor_place.symbol}.",
        sldp_set(sorted_symbols(place_adjacency[neighbor_place.symbol])),
    )
    degree_place = choices.one(scene.places, "a place degree")
    q["qa_place_degree"] = Question(
        f"How many same-layer place edges touch {degree_place.symbol}?",
        str(len(place_adjacency[degree_place.symbol])),
    )
    hop_place = choices.one(connected_places, "a hop-query place")
    hop_budget = choices.one([2, 3], "a hop budget")
    reached = {hop_place.symbol}
    frontier = {hop_place.symbol}
    for _ in range(hop_budget):
        frontier = (
            set().union(*(place_adjacency[symbol] for symbol in frontier)) - reached
        )
        reached.update(frontier)
    reached.discard(hop_place.symbol)
    q["qa_places_within_hops"] = Question(
        f"Which other places are reachable from {hop_place.symbol} in at most {hop_budget} place edges?",
        sldp_set(sorted_symbols(reached)),
    )
    place_left, place_right, place_path = pair_with_path(scene, "place")
    q["qa_place_shortest_path"] = Question(
        f"Trace one minimum-hop place route from {place_left.symbol} to {place_right.symbol}.",
        sldp_list(place_path),
    )
    neighbor_object_places = [
        p
        for p in connected_places
        if any(
            scene.objects_by_place[neighbor] for neighbor in place_adjacency[p.symbol]
        )
    ]
    object_neighbor_place = choices.one(
        neighbor_object_places, "a place neighboring an object-bearing place"
    )
    neighbor_objects = sorted_symbols(
        object_symbol
        for neighbor in place_adjacency[object_neighbor_place.symbol]
        for object_symbol in scene.objects_by_place[neighbor]
    )
    q["qa_objects_in_neighbor_places"] = Question(
        f"Which objects belong to places directly adjacent to {object_neighbor_place.symbol}?",
        sldp_set(neighbor_objects),
    )

    reference = choices.one(scene.objects, "an object distance reference")
    other_objects = [candidate for candidate in scene.objects if candidate != reference]
    q["qa_nearest_object"] = Question(
        f"Using center-to-center distance, which object is nearest to {reference.symbol}?",
        sldp_set(entity.symbol for entity in extrema(reference, other_objects)),
    )
    far_reference = choices.one(scene.objects, "an object far-distance reference")
    q["qa_farthest_object"] = Question(
        f"Which object center lies farthest from {far_reference.symbol}?",
        sldp_set(
            entity.symbol
            for entity in extrema(
                far_reference,
                [
                    candidate
                    for candidate in scene.objects
                    if candidate != far_reference
                ],
                farthest=True,
            )
        ),
    )
    mixed_refs = [
        entity
        for entity in scene.objects
        if any(other.semantic_class != entity.semantic_class for other in scene.objects)
    ]
    mixed_ref = choices.one(mixed_refs, "a different-class distance reference")
    different = [
        entity
        for entity in scene.objects
        if entity.semantic_class != mixed_ref.semantic_class
    ]
    q["qa_nearest_different_class_object"] = Question(
        f"What is the closest object to {mixed_ref.symbol} that has a different semantic class?",
        sldp_set(entity.symbol for entity in extrema(mixed_ref, different)),
    )
    class_a, class_b = choices.two(list(scene.objects_by_class), "two object classes")
    cross_pairs = [
        (euclidean(left, right), left, right)
        for left in scene.objects_by_class[class_a]
        for right in scene.objects_by_class[class_b]
    ]
    cross_distance = min(item[0] for item in cross_pairs)
    closest_cross_pairs = [
        sldp_list(sorted_symbols((left.symbol, right.symbol)))
        for distance_value, left, right in cross_pairs
        if math.isclose(distance_value, cross_distance, abs_tol=1e-9)
    ]
    class_a_word = choices.vocabulary(class_a)
    class_b_word = choices.vocabulary(class_b)
    q["qa_closest_cross_class_pair"] = Question(
        f"Which pairs consisting of {indefinite(class_a_word)} and {indefinite(class_b_word)} are tied for the smallest center distance?",
        sldp_set(sorted(closest_cross_pairs)),
    )
    place_ref = choices.one(scene.objects, "an object for nearest-place lookup")
    q["qa_nearest_place_to_object"] = Question(
        f"Which navigable place center is closest to object {place_ref.symbol}?",
        sldp_set(entity.symbol for entity in extrema(place_ref, scene.places)),
    )
    room_ref = choices.one(scene.objects, "an object for nearest-room lookup")
    q["qa_nearest_room_to_object"] = Question(
        f"Which room center is geometrically closest to {room_ref.symbol}?",
        sldp_set(entity.symbol for entity in extrema(room_ref, scene.rooms)),
    )
    room_pairs = [
        (euclidean(left, right), left, right)
        for left, right in combinations(scene.rooms, 2)
    ]
    if not room_pairs:
        raise GenerationError(
            "Scene needs two rooms for the closest-room-pair question"
        )
    room_pair_distance = min(value for value, _, _ in room_pairs)
    closest_room_pairs = [
        sldp_list(sorted_symbols((left.symbol, right.symbol)))
        for value, left, right in room_pairs
        if math.isclose(value, room_pair_distance, abs_tol=1e-9)
    ]
    q["qa_closest_room_pair"] = Question(
        "Which room pairs are tied for the smallest distance between recorded centers?",
        sldp_set(sorted(closest_room_pairs)),
    )
    height_class = choices.class_name()
    q["qa_highest_object_of_class"] = Question(
        f"Which {choices.vocabulary(height_class)} reaches the greatest center height?",
        sldp_set(
            entity.symbol
            for entity in axis_extrema(scene.objects_by_class[height_class], 2, True)
        ),
    )
    low_class = choices.class_name()
    q["qa_lowest_object_of_class"] = Question(
        f"Which {choices.vocabulary(low_class)} has the lowest center z-coordinate?",
        sldp_set(
            entity.symbol
            for entity in axis_extrema(scene.objects_by_class[low_class], 2, False)
        ),
    )
    near_room = choices.one(scene.rooms, "a room for nearest-object lookup")
    q["qa_nearest_object_to_room"] = Question(
        f"Which object center is nearest to the center of {near_room.symbol}?",
        sldp_set(entity.symbol for entity in extrema(near_room, scene.objects)),
    )
    far_room = choices.one(scene.rooms, "a room for farthest-object lookup")
    q["qa_farthest_object_from_room"] = Question(
        f"Find the object farthest from room center {far_room.symbol}.",
        sldp_set(
            entity.symbol for entity in extrema(far_room, scene.objects, farthest=True)
        ),
    )
    radius_ref = choices.one(scene.objects, "a radius-query object")
    distances = sorted(
        euclidean(radius_ref, candidate)
        for candidate in scene.objects
        if candidate != radius_ref
    )
    if not distances:
        raise GenerationError("Scene needs two objects for the radius question")
    rank = min(4, len(distances)) - 1
    # Ground against the exact threshold printed in the question.  The extra
    # centimeter keeps the selected rank inside after two-decimal rounding.
    radius = round(distances[rank] + 0.01, 2)
    radius_objects = sorted_symbols(
        candidate.symbol
        for candidate in scene.objects
        if candidate != radius_ref and euclidean(radius_ref, candidate) <= radius
    )
    q["qa_objects_within_radius"] = Question(
        f"Which other objects have centers no more than {format_number(radius)} meters from {radius_ref.symbol}?",
        sldp_set(radius_objects),
    )
    return q


def pddl_questions(scene: Scene, choices: Choices) -> dict[str, Question]:
    q: dict[str, Question] = {}
    obj = choices.one(scene.objects, "a PDDL object")
    obj2 = choices.one(
        [item for item in scene.objects if item != obj], "a second PDDL object"
    )
    room = choices.one(scene.rooms, "a PDDL room")
    place = choices.one(scene.places, "a PDDL place")

    q["pddl_visit_object_symbol"] = Question(
        f"Pass by object {obj.symbol} before the task is complete.",
        pddl_fact("visited-object", obj.symbol),
    )
    q["pddl_end_at_object_symbol"] = Question(
        f"Finish the mission at {obj2.symbol}.", pddl_fact("at-object", obj2.symbol)
    )
    inspect_obj = choices.one(scene.objects, "an inspection object")
    q["pddl_inspect_object_symbol"] = Question(
        f"Examine object {inspect_obj.symbol} closely.",
        pddl_fact("safe", inspect_obj.symbol),
    )
    hold_obj = choices.one(scene.objects, "a pickup object")
    q["pddl_hold_object_symbol"] = Question(
        f"Pick up {hold_obj.symbol} and retain it.",
        pddl_fact("holding", hold_obj.symbol),
    )
    q["pddl_visit_room_symbol"] = Question(
        f"Enter room {room.symbol} at some point.",
        pddl_fact("visited-region", room.symbol),
    )
    end_room = choices.one(scene.rooms, "a final room")
    q["pddl_end_in_room_symbol"] = Question(
        f"Conclude the route inside {end_room.symbol}.",
        pddl_fact("in-region", end_room.symbol),
    )
    q["pddl_visit_place_symbol"] = Question(
        f"Include place {place.symbol} on the route.",
        pddl_fact("visited-place", place.symbol),
    )
    end_place = choices.one(scene.places, "a final place")
    q["pddl_end_at_place_symbol"] = Question(
        f"Make {end_place.symbol} the final waypoint.",
        pddl_fact("at-place", end_place.symbol),
    )

    hold_class = choices.class_name(multiple=True)
    hold_members = scene.objects_by_class[hold_class]
    q["pddl_hold_object_class"] = Question(
        f"Bring along any {choices.vocabulary(hold_class)} you can choose.",
        pddl_for_entities("holding", hold_members, force_operator=True),
    )
    visit_class = choices.class_name(multiple=True)
    q["pddl_visit_object_class"] = Question(
        f"Approach one of the {choices.vocabulary(visit_class)} objects.",
        pddl_for_entities(
            "visited-object",
            scene.objects_by_class[visit_class],
            force_operator=True,
        ),
    )
    end_class = choices.class_name(multiple=True)
    end_class_word = choices.vocabulary(end_class)
    q["pddl_end_at_object_class"] = Question(
        f"Stop beside {indefinite(end_class_word)} when finished.",
        pddl_for_entities(
            "at-object", scene.objects_by_class[end_class], force_operator=True
        ),
    )
    inspect_class = choices.class_name(multiple=True)
    q["pddl_inspect_object_class"] = Question(
        f"Check any one {choices.vocabulary(inspect_class)} for safety.",
        pddl_for_entities(
            "safe", scene.objects_by_class[inspect_class], force_operator=True
        ),
    )
    visit_room_class = choices.room_class(multiple=True)
    visit_room_word = choices.vocabulary(visit_room_class)
    q["pddl_visit_room_class"] = Question(
        f"Travel through {indefinite(visit_room_word)} of your choice.",
        pddl_for_entities(
            "visited-region",
            scene.rooms_by_class[visit_room_class],
            force_operator=True,
        ),
    )
    end_room_class = choices.room_class(multiple=True)
    q["pddl_end_in_room_class"] = Question(
        f"End the assignment in any {choices.vocabulary(end_room_class)}.",
        pddl_for_entities(
            "in-region",
            scene.rooms_by_class[end_room_class],
            force_operator=True,
        ),
    )

    objects_with_room = [
        item for item in scene.objects if len(scene.rooms_for_object(item.symbol)) == 1
    ]
    hierarchy_obj = choices.one(objects_with_room, "an object in a room")
    hierarchy_room = scene.rooms_for_object(hierarchy_obj.symbol)[0]
    q["pddl_visit_room_containing_object"] = Question(
        f"Visit the room that contains {hierarchy_obj.symbol}.",
        pddl_fact("visited-region", hierarchy_room),
    )
    hierarchy_end_obj = choices.one(objects_with_room, "another object in a room")
    q["pddl_end_in_room_containing_object"] = Question(
        f"Finish in whichever room contains object {hierarchy_end_obj.symbol}.",
        pddl_fact("in-region", scene.rooms_for_object(hierarchy_end_obj.symbol)[0]),
    )
    objects_with_place = [
        item
        for item in scene.objects
        if len(item.parents) == 1 and item.parent in scene.entities
    ]
    parent_obj = choices.one(objects_with_place, "an object with a place")
    q["pddl_visit_parent_place"] = Question(
        f"Navigate through the place holding {parent_obj.symbol}.",
        pddl_fact("visited-place", parent_obj.parent or ""),
    )
    parent_end_obj = choices.one(objects_with_place, "an object with a final place")
    q["pddl_end_at_parent_place"] = Question(
        f"Finish at the place directly containing {parent_end_obj.symbol}.",
        pddl_fact("at-place", parent_end_obj.parent or ""),
    )

    near_ref = choices.one(scene.objects, "a nearest-object reference")
    near_targets = extrema(
        near_ref, [item for item in scene.objects if item != near_ref]
    )
    q["pddl_visit_nearest_object"] = Question(
        f"Visit the object whose center is nearest to {near_ref.symbol}.",
        pddl_for_entities("visited-object", near_targets),
    )
    inspect_ref = choices.one(scene.objects, "a nearest inspection reference")
    inspect_targets = extrema(
        inspect_ref, [item for item in scene.objects if item != inspect_ref]
    )
    q["pddl_inspect_nearest_object"] = Question(
        f"Inspect the object positioned closest to {inspect_ref.symbol}.",
        pddl_for_entities("safe", inspect_targets),
    )
    nearest_class = choices.class_name(multiple=True)
    nearest_class_ref = choices.one(
        [item for item in scene.objects if item.semantic_class != nearest_class],
        "a reference outside a target class",
    )
    nearest_class_targets = extrema(
        nearest_class_ref, scene.objects_by_class[nearest_class]
    )
    q["pddl_hold_nearest_class_object"] = Question(
        f"Pick up the {choices.vocabulary(nearest_class)} closest to {nearest_class_ref.symbol}.",
        pddl_for_entities("holding", nearest_class_targets),
    )
    far_ref = choices.one(scene.objects, "a farthest-object reference")
    far_targets = extrema(
        far_ref, [item for item in scene.objects if item != far_ref], farthest=True
    )
    q["pddl_visit_farthest_object"] = Question(
        f"Reach the object farthest from {far_ref.symbol} by center distance.",
        pddl_for_entities("visited-object", far_targets),
    )
    far_end_ref = choices.one(scene.objects, "a farthest final reference")
    far_end_targets = extrema(
        far_end_ref,
        [item for item in scene.objects if item != far_end_ref],
        farthest=True,
    )
    q["pddl_end_at_farthest_object"] = Question(
        f"Finish at the object most distant from {far_end_ref.symbol}.",
        pddl_for_entities("at-object", far_end_targets),
    )
    high_class = choices.class_name()
    q["pddl_inspect_highest_class_object"] = Question(
        f"Inspect the highest-positioned {choices.vocabulary(high_class)}.",
        pddl_for_entities(
            "safe", axis_extrema(scene.objects_by_class[high_class], 2, True)
        ),
    )
    low_class = choices.class_name()
    q["pddl_visit_lowest_class_object"] = Question(
        f"Visit the {choices.vocabulary(low_class)} with the smallest center height.",
        pddl_for_entities(
            "visited-object", axis_extrema(scene.objects_by_class[low_class], 2, False)
        ),
    )
    nearest_room_ref = choices.one(scene.objects, "a nearest-room reference")
    nearest_rooms = extrema(nearest_room_ref, scene.rooms)
    q["pddl_visit_nearest_room"] = Question(
        f"Enter the room whose center is closest to {nearest_room_ref.symbol}.",
        pddl_for_entities("visited-region", nearest_rooms),
    )
    nearest_end_ref = choices.one(scene.objects, "a nearest final-room reference")
    q["pddl_end_in_nearest_room"] = Question(
        f"Conclude in the room geometrically nearest to {nearest_end_ref.symbol}.",
        pddl_for_entities("in-region", extrema(nearest_end_ref, scene.rooms)),
    )

    room_adjacency = scene.adjacency("room")
    max_degree = max(map(len, room_adjacency.values()))
    min_degree = min(map(len, room_adjacency.values()))
    q["pddl_visit_most_connected_room"] = Question(
        "Visit a room with the maximum number of direct room connections.",
        pddl_group(
            "or",
            [
                pddl_fact("visited-region", symbol)
                for symbol, links in room_adjacency.items()
                if len(links) == max_degree
            ],
        ),
    )
    q["pddl_end_in_least_connected_room"] = Question(
        "Finish in a room tied for the smallest connectivity degree.",
        pddl_group(
            "or",
            [
                pddl_fact("in-region", symbol)
                for symbol, links in room_adjacency.items()
                if len(links) == min_degree
            ],
        ),
    )
    rooms_with_neighbors = [item for item in scene.rooms if room_adjacency[item.symbol]]
    adjacent_ref = choices.one(rooms_with_neighbors, "a room with adjacent rooms")
    q["pddl_visit_adjacent_rooms"] = Question(
        f"Cover every room directly connected to {adjacent_ref.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-region", symbol)
                for symbol in sorted_symbols(room_adjacency[adjacent_ref.symbol])
            ],
            always=True,
        ),
    )

    either_a, either_b = choices.two(scene.objects, "an object disjunction")
    q["pddl_visit_either_object"] = Question(
        f"Visit either {either_a.symbol} or {either_b.symbol}; one is enough.",
        pddl_group(
            "or",
            [
                pddl_fact("visited-object", either_a.symbol),
                pddl_fact("visited-object", either_b.symbol),
            ],
        ),
    )
    cross_obj = choices.one(scene.objects, "a cross-layer object")
    cross_room = choices.one(scene.rooms, "a cross-layer room")
    q["pddl_visit_object_or_room"] = Question(
        f"Either pass object {cross_obj.symbol} or enter room {cross_room.symbol}.",
        pddl_group(
            "or",
            [
                pddl_fact("visited-object", cross_obj.symbol),
                pddl_fact("visited-region", cross_room.symbol),
            ],
        ),
    )
    both_obj = choices.one(scene.objects, "a conjunctive object")
    both_room = choices.one(scene.rooms, "a conjunctive room")
    q["pddl_visit_object_and_room"] = Question(
        f"Make sure you pass {both_obj.symbol} and also enter {both_room.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-object", both_obj.symbol),
                pddl_fact("visited-region", both_room.symbol),
            ],
        ),
    )
    inspect_hold = choices.one(scene.objects, "an inspect-and-hold object")
    q["pddl_inspect_and_hold_object"] = Question(
        f"Verify {inspect_hold.symbol} is safe and keep hold of it.",
        pddl_group(
            "and",
            [
                pddl_fact("safe", inspect_hold.symbol),
                pddl_fact("holding", inspect_hold.symbol),
            ],
        ),
    )

    order_a, order_b = choices.two(scene.objects, "an ordered object pair")
    q["pddl_visit_then_end_at_object"] = Question(
        f"Pass {order_a.symbol} first, then finish at {order_b.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-object", order_a.symbol),
                pddl_fact("at-object", order_b.symbol),
            ],
        ),
    )
    order_room = choices.one(scene.rooms, "an ordered first room")
    order_object = choices.one(scene.objects, "an ordered final object")
    q["pddl_visit_room_then_end_at_object"] = Question(
        f"Travel through {order_room.symbol} before ending at {order_object.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-region", order_room.symbol),
                pddl_fact("at-object", order_object.symbol),
            ],
        ),
    )
    first_object = choices.one(scene.objects, "an ordered first object")
    final_room = choices.one(scene.rooms, "an ordered final room")
    q["pddl_visit_object_then_end_in_room"] = Question(
        f"Visit {first_object.symbol}, and make {final_room.symbol} your final room.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-object", first_object.symbol),
                pddl_fact("in-region", final_room.symbol),
            ],
        ),
    )
    first_place = choices.one(scene.places, "an ordered first place")
    place_final_room = choices.one(scene.rooms, "a place-to-room final room")
    q["pddl_visit_place_then_end_in_room"] = Question(
        f"Go via {first_place.symbol}, then end inside {place_final_room.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-place", first_place.symbol),
                pddl_fact("in-region", place_final_room.symbol),
            ],
        ),
    )
    first_room = choices.one(scene.rooms, "a room-to-place first room")
    final_place = choices.one(scene.places, "a room-to-place final place")
    q["pddl_visit_room_then_end_at_place"] = Question(
        f"Enter {first_room.symbol} along the way and stop finally at {final_place.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-region", first_room.symbol),
                pddl_fact("at-place", final_place.symbol),
            ],
        ),
    )
    place_a, place_b = choices.two(scene.places, "an ordered place pair")
    q["pddl_visit_then_end_at_place"] = Question(
        f"Include {place_a.symbol} before making {place_b.symbol} the endpoint.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-place", place_a.symbol),
                pddl_fact("at-place", place_b.symbol),
            ],
        ),
    )
    inspect_order_obj = choices.one(scene.objects, "an inspection/order object")
    inspect_order_place = choices.one(scene.places, "an inspection/order place")
    q["pddl_inspect_then_end_at_place"] = Question(
        f"Inspect {inspect_order_obj.symbol}, then conclude at {inspect_order_place.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("safe", inspect_order_obj.symbol),
                pddl_fact("at-place", inspect_order_place.symbol),
            ],
        ),
    )
    carry_obj = choices.one(scene.objects, "a carried object")
    carry_room = choices.one(scene.rooms, "a carrying destination room")
    q["pddl_hold_then_end_in_room"] = Question(
        f"Keep {carry_obj.symbol} in hand and finish inside {carry_room.symbol}.",
        pddl_group(
            "and",
            [
                pddl_fact("holding", carry_obj.symbol),
                pddl_fact("in-region", carry_room.symbol),
            ],
        ),
    )

    move_obj = choices.one(objects_with_place, "an object to relocate")
    move_place = choices.one(
        [item for item in scene.places if item.symbol != move_obj.parent],
        "a relocation destination",
    )
    q["pddl_move_object_to_place"] = Question(
        f"Relocate {move_obj.symbol} into place {move_place.symbol}.",
        pddl_fact("object-in-place", move_obj.symbol, move_place.symbol),
    )
    move_class = choices.class_name(multiple=True)
    move_class_place = choices.one(scene.places, "a class relocation destination")
    q["pddl_move_class_object_to_place"] = Question(
        f"Move any one {choices.vocabulary(move_class)} into {move_class_place.symbol}.",
        pddl_group(
            "or",
            [
                pddl_fact("object-in-place", member.symbol, move_class_place.symbol)
                for member in scene.objects_by_class[move_class]
            ],
            always=True,
        ),
    )
    place_adjacency = scene.adjacency("place")
    move_neighbor_candidates = [
        item
        for item in objects_with_place
        if place_adjacency.get(item.parent or "", set())
    ]
    neighbor_move_obj = choices.one(
        move_neighbor_candidates, "an object whose place has a neighbor"
    )
    neighbor_destination = choices.one(
        sorted_symbols(place_adjacency[neighbor_move_obj.parent or ""]),
        "a neighboring destination place",
    )
    q["pddl_move_object_to_neighbor_place"] = Question(
        f"Transfer {neighbor_move_obj.symbol} to place {neighbor_destination}, adjacent to its current place.",
        pddl_fact(
            "object-in-place",
            neighbor_move_obj.symbol,
            neighbor_destination,
        ),
    )

    avoid_obj = choices.one(scene.objects, "an object not to hold")
    q["pddl_do_not_hold_object"] = Question(
        f"Ensure object {avoid_obj.symbol} is not being carried.",
        f"(not {pddl_fact('holding', avoid_obj.symbol)})",
    )
    visit_free_obj = choices.one(scene.objects, "an object to visit without carrying")
    q["pddl_visit_without_holding"] = Question(
        f"Visit {visit_free_obj.symbol} but do not take it with you.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-object", visit_free_obj.symbol),
                f"(not {pddl_fact('holding', visit_free_obj.symbol)})",
            ],
        ),
    )
    target_room = choices.one(rooms_with_neighbors, "a room with an avoidable neighbor")
    avoided_room = choices.one(
        sorted_symbols(room_adjacency[target_room.symbol]), "an adjacent room to avoid"
    )
    q["pddl_visit_room_avoid_neighbor"] = Question(
        f"Visit {target_room.symbol} while never entering its neighbor {avoided_room}.",
        pddl_group(
            "and",
            [
                pddl_fact("visited-region", target_room.symbol),
                f"(not {pddl_fact('visited-region', avoided_room)})",
            ],
        ),
    )
    alternative_class = choices.class_name(multiple=True)
    alternative_obj = choices.one(scene.objects, "an alternative inspection object")
    holding_alternatives = pddl_for_entities(
        "holding",
        scene.objects_by_class[alternative_class],
        force_operator=True,
    )
    q["pddl_hold_class_or_inspect_object"] = Question(
        f"Either carry {indefinite(choices.vocabulary(alternative_class))} or inspect {alternative_obj.symbol}.",
        pddl_group(
            "or", [holding_alternatives, pddl_fact("safe", alternative_obj.symbol)]
        ),
    )
    combined_class = choices.class_name(multiple=True)
    combined_room = choices.one(scene.rooms, "a final room for a class visit")
    q["pddl_visit_class_and_end_in_room"] = Question(
        f"Visit {indefinite(choices.vocabulary(combined_class))}, then finish in {combined_room.symbol}.",
        pddl_group(
            "and",
            [
                pddl_for_entities(
                    "visited-object",
                    scene.objects_by_class[combined_class],
                    force_operator=True,
                ),
                pddl_fact("in-region", combined_room.symbol),
            ],
        ),
    )
    return q


def load_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)
    if not isinstance(catalog, dict):
        raise GenerationError("Question type catalog must be a YAML mapping")
    for family in ("qa", "pddl"):
        entries = catalog.get(family)
        if not isinstance(entries, list) or len(entries) != 50:
            raise GenerationError(
                f"Catalog section `{family}` must contain exactly 50 entries"
            )
        ids = [entry.get("id") for entry in entries]
        if len(ids) != len(set(ids)):
            raise GenerationError(f"Catalog section `{family}` contains duplicate IDs")
    return catalog


def materialize(
    family: str,
    catalog: list[dict[str, Any]],
    built: dict[str, Question],
) -> list[dict[str, Any]]:
    catalog_ids = {entry["id"] for entry in catalog}
    if catalog_ids != set(built):
        missing = sorted(catalog_ids - set(built))
        extra = sorted(set(built) - catalog_ids)
        raise GenerationError(
            f"{family} catalog/builder mismatch; missing={missing}, extra={extra}"
        )
    comparator = "SLDP" if family == "qa" else "PDDL"
    prefix = "QA" if family == "qa" else "PDDL"
    records = []
    for index, entry in enumerate(catalog, 1):
        item = built[entry["id"]]
        records.append(
            {
                "uid": f"{prefix}{index:03d}",
                "name": entry["name"],
                "question": item.question,
                "solution": item.solution,
                "correctness_comparator": {
                    "comparison_type": comparator,
                    "relation": "equal",
                },
                "tags": list(entry.get("tags", [])),
            }
        )
    return records


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def questions_from_files(paths: Iterable[Path]) -> dict[str, Path]:
    questions: dict[str, Path] = {}

    def extract(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"question", "questions", "question_variants"}:
                    if isinstance(child, str):
                        yield child
                    elif isinstance(child, list):
                        yield from (item for item in child if isinstance(item, str))
                yield from extract(child)
        elif isinstance(value, list):
            for child in value:
                yield from extract(child)

    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        for text in extract(data):
            questions[normalize_question(text)] = path
    return questions


def validate_questions(
    qa_records: list[dict[str, Any]],
    pddl_records: list[dict[str, Any]],
    avoid_paths: list[Path],
) -> None:
    all_records = qa_records + pddl_records
    normalized = [normalize_question(record["question"]) for record in all_records]
    if len(normalized) != len(set(normalized)):
        raise GenerationError("Generated natural-language questions are not unique")
    known = questions_from_files(avoid_paths)
    collisions = [
        record["question"]
        for record in all_records
        if normalize_question(record["question"]) in known
    ]
    if collisions:
        raise GenerationError(
            "Generated questions exactly match denylisted examples: " + repr(collisions)
        )


def write_yaml(
    path: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(
            {"metadata": metadata, "questions": records},
            stream,
            Dumper=NoAliasDumper,
            allow_unicode=True,
            sort_keys=False,
            explicit_start=True,
            width=100,
        )


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(
            metadata,
            stream,
            Dumper=NoAliasDumper,
            allow_unicode=True,
            sort_keys=False,
            explicit_start=True,
            width=100,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_graph",
        nargs="?",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"Spark DSG JSON file (default: {DEFAULT_SCENE.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="YAML catalog defining the 50 QA and 50 PDDL types",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for question files and metadata. Defaults to "
            "data/questions/<scene-id>."
        ),
    )
    parser.add_argument(
        "--scene-id",
        help=(
            "Stable identifier used in metadata and as the default output folder; "
            "defaults to the scene filename stem"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed controlling grounded entity and vocabulary choices (default: 7)",
    )
    parser.add_argument(
        "--avoid-questions",
        action="append",
        type=Path,
        default=[],
        metavar="YAML",
        help="Reject exact question-text collisions with this YAML file; repeatable",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Print the catalog IDs without loading a scene graph",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if yaml is None:
        print(
            "error: PyYAML is required; install it with `pip install PyYAML`",
            file=sys.stderr,
        )
        return 2
    try:
        catalog = load_catalog(args.catalog)
        if args.list_types:
            for family in ("qa", "pddl"):
                print(f"{family.upper()} ({len(catalog[family])})")
                for entry in catalog[family]:
                    print(f"  {entry['id']}: {entry['description']}")
            return 0
        if spark_dsg is None:
            print(
                "error: spark-dsg is required; install it with `pip install spark-dsg`",
                file=sys.stderr,
            )
            return 2
        if not args.scene_graph.is_file():
            raise GenerationError(f"Scene graph does not exist: {args.scene_graph}")
        if not args.catalog.is_file():
            raise GenerationError(f"Question catalog does not exist: {args.catalog}")
        scene = Scene.load(args.scene_graph)
        if not scene.objects or not scene.places or not scene.rooms:
            raise GenerationError(
                "Scene must contain Objects, Rooms, and at least one 2D or 3D Places layer"
            )

        qa = materialize(
            "qa", catalog["qa"], qa_questions(scene, Choices(scene, args.seed))
        )
        pddl = materialize(
            "pddl",
            catalog["pddl"],
            pddl_questions(scene, Choices(scene, args.seed + 1)),
        )

        default_examples = [
            ROOT
            / "external"
            / "heracles_agents"
            / "examples"
            / "questions"
            / "qa_questions.yaml",
            ROOT
            / "external"
            / "heracles_agents"
            / "examples"
            / "questions"
            / "pddl_questions.yaml",
        ]
        avoid_paths = list(args.avoid_questions)
        avoid_paths.extend(path for path in default_examples if path.is_file())
        validate_questions(qa, pddl, avoid_paths)

        scene_id = args.scene_id or re.sub(
            r"[^A-Za-z0-9_.-]+", "_", args.scene_graph.stem
        ).strip("_")
        if not scene_id:
            raise GenerationError(
                "Scene graph filename cannot produce a valid scene ID"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", scene_id):
            raise GenerationError(
                "Scene ID may contain only letters, digits, '.', '_', and '-'"
            )
        output_dir = args.output_dir or DEFAULT_QUESTION_ROOT / scene_id
        if output_dir.name != scene_id:
            raise GenerationError(
                "The output directory name must match the scene ID so question "
                "sets remain grouped by source scene"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        shared_metadata = {
            "schema_version": 1,
            "scene_id": scene_id,
            "scene_graph": {
                "path": display_path(args.scene_graph),
                "sha256": file_sha256(args.scene_graph),
            },
            "question_catalog": {
                "path": display_path(args.catalog),
                "sha256": file_sha256(args.catalog),
            },
            "generator": {
                "path": display_path(Path(__file__)),
                "version": GENERATOR_VERSION,
                "sha256": file_sha256(Path(__file__)),
                "spark_dsg_version": package_version("spark-dsg"),
                "pyyaml_version": package_version("PyYAML"),
            },
            "parameters": {
                "seed": args.seed,
                "qa_seed": args.seed,
                "pddl_seed": args.seed + 1,
                "qa_question_count": len(qa),
                "pddl_question_count": len(pddl),
                "avoid_question_files": [
                    {
                        "path": display_path(path),
                        "sha256": file_sha256(path),
                    }
                    for path in avoid_paths
                ],
            },
        }

        qa_path = output_dir / "qa_questions.yaml"
        pddl_path = output_dir / "pddl_questions.yaml"
        write_yaml(qa_path, qa, {**shared_metadata, "task": "qa"})
        write_yaml(pddl_path, pddl, {**shared_metadata, "task": "pddl"})

        metadata_path = output_dir / "metadata.yaml"
        write_metadata(
            metadata_path,
            {
                **shared_metadata,
                "artifacts": {
                    "qa": {
                        "path": display_path(qa_path),
                        "sha256": file_sha256(qa_path),
                    },
                    "pddl": {
                        "path": display_path(pddl_path),
                        "sha256": file_sha256(pddl_path),
                    },
                },
            },
        )
        print(f"Wrote {len(qa)} QA questions to {qa_path}")
        print(f"Wrote {len(pddl)} PDDL questions to {pddl_path}")
        print(f"Wrote reproduction metadata to {metadata_path}")
        return 0
    except (GenerationError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

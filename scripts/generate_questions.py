#!/usr/bin/env python3
"""Generate deterministic QA and PDDL benchmark questions from a Spark DSG.

The generator creates one grounded question for each supported type declared in
``data/questions/question_types_v2.yaml``: 50 scene-graph QA questions and 50 PDDL
goals. Entity choices and wording are reproducible for a given random seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
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
DEFAULT_SCENE = ROOT / "data" / "scene_graphs" / "spark_dsg_example.json"
DEFAULT_CATALOG = ROOT / "data" / "questions" / "question_types_v2.yaml"
DEFAULT_QUESTION_ROOT = ROOT / "data" / "questions"
GENERATOR_VERSION = 9
PAIR_QA_FAMILIES = frozenset(
    {
        "qa_closest_cross_class_pair",
        "qa_closest_room_pair",
        "qa_closest_class_pairs_in_room_class",
        "qa_closest_pair_in_different_rooms",
        "qa_mutual_nearest_class_pairs",
        "qa_closest_class_pairs_per_room",
    }
)
OVERLAP_CLASSES = frozenset({"direct", "related", "absent"})
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
    parameters: dict[str, Any]
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
        self._index()

    @classmethod
    def from_entities(
        cls, entities: Iterable[Entity], metadata: dict | None = None
    ) -> Scene:
        """Build an index from a typed scene adapter or an in-memory test graph."""
        scene = cls.__new__(cls)
        scene.graph = None
        scene._metadata = metadata or {}
        entities = list(entities)
        scene.objects = [e for e in entities if e.kind == "object"]
        scene.places = [e for e in entities if e.kind in {"place", "mesh_place"}]
        scene.rooms = [e for e in entities if e.kind == "room"]
        for values in (scene.objects, scene.places, scene.rooms):
            values.sort(key=lambda e: symbol_key(e.symbol))
        scene._index()
        return scene

    def _index(self) -> None:
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
        obj = self.entities.get(object_symbol)
        if obj is None or obj.kind != "object":
            return []
        return self.rooms_for_entity(object_symbol)

    def rooms_for_entity(self, symbol: str) -> list[str]:
        obj = self.entities.get(symbol)
        if obj is None:
            return []
        rooms: set[str] = set()
        visited: set[str] = set()
        frontier = list(obj.parents)
        while frontier:
            symbol = frontier.pop()
            if symbol in visited or symbol not in self.entities:
                continue
            visited.add(symbol)
            parent = self.entities[symbol]
            if parent.kind == "room":
                rooms.add(symbol)
            elif parent.kind in {"place", "mesh_place"}:
                frontier.extend(parent.parents)
        return sorted_symbols(rooms)

    def adjacency(self, kind: str) -> dict[str, set[str]]:
        entities = self.rooms if kind == "room" else self.places
        valid = {entity.symbol for entity in entities}
        adjacency = {entity.symbol: set() for entity in entities}
        by_id = {entity.symbol: entity for entity in entities}
        for entity in entities:
            for neighbor in set(entity.neighbors).intersection(valid):
                if neighbor != entity.symbol and by_id[neighbor].kind == entity.kind:
                    adjacency[entity.symbol].add(neighbor)
                    adjacency[neighbor].add(entity.symbol)
        return adjacency

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


def load_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    from scripts.question_sampling import CAPABILITIES, PARAMETER_TYPES

    with path.open("r", encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)
    if not isinstance(catalog, dict):
        raise GenerationError("Question type catalog must be a YAML mapping")
    if set(catalog) != {"qa", "pddl"}:
        raise GenerationError("Catalog must contain only qa and pddl lists")
    for family in ("qa", "pddl"):
        entries = catalog.get(family)
        if not isinstance(entries, list) or len(entries) != 50:
            raise GenerationError(
                f"Catalog section `{family}` must contain exactly 50 entries"
            )
        if any(not isinstance(entry, dict) for entry in entries):
            raise GenerationError(
                f"Catalog section `{family}` entries must be mappings"
            )
        ids = [entry.get("id") for entry in entries]
        if any(
            not isinstance(identifier, str) or not identifier.startswith(f"{family}_")
            for identifier in ids
        ):
            raise GenerationError(
                f"Catalog section `{family}` IDs must start with `{family}_`"
            )
        if len(ids) != len(set(ids)):
            raise GenerationError(f"Catalog section `{family}` contains duplicate IDs")
        for entry in entries:
            expected_fields = {
                "id",
                "description",
                "question_template",
                "overlap_class",
                "tags",
                "parameters",
                "requires",
            }
            if set(entry) != expected_fields:
                raise GenerationError(
                    f"{entry['id']}: expected fields {sorted(expected_fields)}"
                )
            template = entry.get("question_template")
            if not isinstance(template, str) or not template.strip():
                raise GenerationError(
                    f"{entry['id']}: question_template must be non-empty text"
                )
            if entry.get("overlap_class") not in OVERLAP_CLASSES:
                raise GenerationError(
                    f"{entry['id']}: overlap_class must be direct, related, or absent"
                )
            try:
                fields = list(Formatter().parse(template))
            except ValueError as exc:
                raise GenerationError(
                    f"{entry['id']}: invalid question_template: {exc}"
                ) from exc
            for _, field, spec, conversion in fields:
                if field is not None and (
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field)
                    or spec
                    or conversion
                ):
                    raise GenerationError(
                        f"{entry['id']}: use simple named placeholders such as {{object}}"
                    )
            parameters = entry.get("parameters")
            names = {field for _, field, _, _ in fields if field is not None}
            if not isinstance(parameters, dict) or set(parameters) != names:
                raise GenerationError(
                    f"{entry['id']}: parameters must declare exactly {sorted(names)}"
                )
            if any(
                not isinstance(value, str) or value not in PARAMETER_TYPES
                for value in parameters.values()
            ):
                raise GenerationError(f"{entry['id']}: unknown parameter type")
            if (
                "object_class_counted" in parameters.values()
                and parameters.get("required_count") != "positive_integer"
            ):
                raise GenerationError(
                    f"{entry['id']}: counted nouns require a positive_integer required_count"
                )
            requires = entry.get("requires")
            if not isinstance(requires, list) or any(
                not isinstance(value, str) or value not in CAPABILITIES
                for value in requires
            ):
                raise GenerationError(
                    f"{entry['id']}: requires must list known capabilities"
                )
            if (
                not isinstance(entry.get("description"), str)
                or not entry["description"].strip()
                or not isinstance(entry.get("tags"), list)
                or any(not isinstance(tag, str) for tag in entry["tags"])
            ):
                raise GenerationError(
                    f"{entry['id']}: description and tags are required"
                )
    return catalog


def materialize(
    family: str,
    catalog: list[dict[str, Any]],
    built: dict[str, Question],
    *,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
    catalog_ids = {entry["id"] for entry in catalog}
    if set(built) - catalog_ids or (not allow_partial and catalog_ids != set(built)):
        missing = sorted(catalog_ids - set(built))
        extra = sorted(set(built) - catalog_ids)
        raise GenerationError(
            f"{family} catalog/builder mismatch; missing={missing}, extra={extra}"
        )
    comparator = "SLDP" if family == "qa" else "PDDL"
    prefix = "QA" if family == "qa" else "PDDL"
    records = []
    for index, entry in enumerate(catalog, 1):
        if entry["id"] not in built:
            continue
        item = built[entry["id"]]
        try:
            question_text = entry["question_template"].format_map(item.parameters)
        except (KeyError, ValueError) as exc:
            raise GenerationError(
                f"{entry['id']}: cannot render question_template: {exc}"
            ) from exc
        records.append(
            {
                "uid": f"{prefix}{index:03d}",
                "name": entry["id"]
                .removeprefix(f"{family}_")
                .replace("_", " ")
                .capitalize(),
                "question_type": entry["id"],
                "overlap_class": entry["overlap_class"],
                "question": question_text,
                "solution": item.solution,
                "correctness_comparator": {
                    "comparison_type": comparator,
                    "relation": "equal",
                    **(
                        {"unordered_pairs": True}
                        if entry["id"] in PAIR_QA_FAMILIES
                        else {}
                    ),
                },
                "tags": list(entry.get("tags", [])),
            }
        )
    return records


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def question_pattern(value: str) -> re.Pattern[str]:
    """Match wording while treating training-template slots as arbitrary values."""
    parts = re.split(r"(\{[A-Za-z_][A-Za-z0-9_.]*\})", normalize_question(value))
    return re.compile(
        "".join(
            r".+?"
            if re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_.]*\}", part)
            else re.escape(part)
            for part in parts
        )
    )


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
                if key in {
                    "question",
                    "questions",
                    "question_variants",
                    "question_template",
                }:
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
    catalog: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    all_records = qa_records + pddl_records
    normalized = [normalize_question(record["question"]) for record in all_records]
    if len(normalized) != len(set(normalized)):
        raise GenerationError("Generated natural-language questions are not unique")
    known = questions_from_files(avoid_paths)
    patterns = [question_pattern(template) for template in known]
    candidates = [record["question"] for record in all_records]
    if catalog is not None:
        candidates.extend(
            entry["question_template"]
            for family in ("qa", "pddl")
            for entry in catalog[family]
        )
    collisions = [
        question
        for question in candidates
        if any(pattern.fullmatch(normalize_question(question)) for pattern in patterns)
    ]
    if collisions:
        raise GenerationError(
            "Generated questions match denylisted wording or templates: "
            + repr(collisions)
        )


def write_yaml(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(
            {"questions": records},
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
            "data/questions/<catalog-version>/<scene-id>."
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
        help="Reject wording collisions, including parameterized templates, from this YAML; repeatable",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Print the catalog IDs without loading a scene graph",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=48,
        help="Eligible candidates retained per family for diversity selection",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=30000,
        help="Maximum evaluated assignments per family; exhaustion is reported",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail without writing a dataset unless all 100 families are supported",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=3,
        help="Also write this many supported questions per task as a smoke suite; 0 disables",
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
        from scripts.question_sampling import Sampler

        if args.smoke_size < 0:
            raise GenerationError("smoke-size must be nonnegative")
        sampler = Sampler(scene, args.seed, args.pool_size, args.max_attempts)
        built = sampler.select(catalog)
        unavailable = {
            key: value
            for key, value in sampler.report["families"].items()
            if value["status"] != "generated"
        }
        if args.strict and unavailable:
            raise GenerationError(
                "Unsupported families in strict mode: " + ", ".join(unavailable)
            )
        qa = materialize("qa", catalog["qa"], built["qa"], allow_partial=True)
        pddl = materialize("pddl", catalog["pddl"], built["pddl"], allow_partial=True)
        if not qa or not pddl:
            raise GenerationError(
                "Scene produced no eligible questions for one or both tasks"
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
        training_templates = (
            ROOT.parent / "3dsg-cypher-agent" / "configs" / "question_templates"
        )
        avoid_paths.extend(
            path
            for path in (
                training_templates / "cypher_qa_3dsg.yaml",
                training_templates / "cypher_pddl_3dsg.yaml",
            )
            if path.is_file()
        )
        avoid_paths = list(dict.fromkeys(avoid_paths))
        validate_questions(qa, pddl, avoid_paths, catalog)

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
        catalog_name = args.catalog.stem.removeprefix("question_types_")
        output_dir = args.output_dir or DEFAULT_QUESTION_ROOT / catalog_name / scene_id
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
                "sampler_path": display_path(
                    Path(__file__).with_name("question_sampling.py")
                ),
                "sampler_sha256": file_sha256(
                    Path(__file__).with_name("question_sampling.py")
                ),
                "spark_dsg_version": package_version("spark-dsg"),
                "pyyaml_version": package_version("PyYAML"),
            },
            "parameters": {
                "seed": args.seed,
                "qa_seed": args.seed,
                "pddl_seed": args.seed,
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
            "sampling": sampler.report,
            "coverage": {
                family: {
                    "requested": len(catalog[family]),
                    "generated": len(built[family]),
                    "unavailable": len(catalog[family]) - len(built[family]),
                }
                for family in ("qa", "pddl")
            },
        }

        qa_path = output_dir / "qa_questions.yaml"
        pddl_path = output_dir / "pddl_questions.yaml"
        write_yaml(qa_path, qa)
        write_yaml(pddl_path, pddl)

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
        if args.smoke_size:
            smoke_qa, smoke_pddl = qa[: args.smoke_size], pddl[: args.smoke_size]
            smoke_qa_path, smoke_pddl_path = (
                output_dir / "qa_questions.smoke.yaml",
                output_dir / "pddl_questions.smoke.yaml",
            )
            write_yaml(smoke_qa_path, smoke_qa)
            write_yaml(smoke_pddl_path, smoke_pddl)
            write_metadata(
                output_dir / "metadata.smoke.yaml",
                {
                    **shared_metadata,
                    "parameters": {
                        **shared_metadata["parameters"],
                        "qa_question_count": len(smoke_qa),
                        "pddl_question_count": len(smoke_pddl),
                    },
                    "coverage": {
                        family: {
                            **shared_metadata["coverage"][family],
                            "selected_for_smoke": len(records),
                        }
                        for family, records in (("qa", smoke_qa), ("pddl", smoke_pddl))
                    },
                    "artifacts": {
                        family: {
                            "path": display_path(path),
                            "sha256": file_sha256(path),
                        }
                        for family, path in (
                            ("qa", smoke_qa_path),
                            ("pddl", smoke_pddl_path),
                        )
                    },
                },
            )
        for identifier, details in unavailable.items():
            print(f"{details['status']}: {identifier}: {details['reason']}")
        return 0
    except (GenerationError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("scripts.generate_questions", sys.modules[__name__])
    raise SystemExit(main())

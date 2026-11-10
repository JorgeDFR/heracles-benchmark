#!/usr/bin/env python3
"""Load benchmark scene labels through the unmodified Heracles API."""

import argparse
import os
import tempfile
from pathlib import Path

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import get_labelspace, load_dsg_to_db


def embedded_labels(graph, keys):
    labelspaces = (graph.metadata.get() or {}).get("labelspaces", {})
    for key in keys:
        if labelspaces.get(key):
            return {int(label): str(name) for label, name in labelspaces[key]}
    return {}


def select_labels(graph, filename, keys, fallback):
    if filename is not None:
        labels = get_labelspace(filename)
    else:
        labels = embedded_labels(graph, keys) or get_labelspace(fallback)
    return {int(label): str(name) for label, name in labels.items()}


def prepare_labels(graph, object_labelspace=None, room_labelspace=None):
    objects = select_labels(
        graph,
        object_labelspace,
        ("_l2p0", "objects", "object_labelspace"),
        "ade20k_mit_label_space.yaml",
    )
    rooms = select_labels(
        graph,
        room_labelspace,
        ("room_labelspace", "_l4p0", "rooms"),
        "b45_label_space.yaml",
    )
    mesh = embedded_labels(graph, ("_l3p1", "mesh", "mesh_places"))
    try:
        mesh_layer = graph.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except IndexError:
        mesh_layer = graph.get_layer(20)

    # Heracles shares one class table between objects and mesh places. Allocate
    # distinct mesh IDs in this in-memory graph when the source taxonomies clash.
    combined = dict(objects)
    mesh_ids = {}
    next_id = max(combined.keys() | mesh.keys(), default=-1) + 1
    for label, name in sorted(mesh.items()):
        target = label
        if label in combined and combined[label] != name:
            target = next_id
            next_id += 1
        combined[target] = name
        mesh_ids[label] = target
    for node in mesh_layer.nodes:
        label = int(node.attributes.semantic_label)
        if mesh:
            if label not in mesh_ids:
                raise ValueError(f"Missing mesh-place class label for node {node.id.str(True)}")
            node.attributes.semantic_label = mesh_ids[label]

    # Validate before Heracles clears or writes to the database.
    for layer, labels in (
        (graph.get_layer(spark_dsg.DsgLayers.OBJECTS), objects),
        (mesh_layer, combined),
        (graph.get_layer(spark_dsg.DsgLayers.ROOMS), rooms),
    ):
        for node in layer.nodes:
            if int(node.attributes.semantic_label) not in labels:
                raise ValueError(f"Missing class label for node {node.id.str(True)}")
    return combined, rooms


def load_scene(scene_path, uri, credentials, object_labelspace=None, room_labelspace=None):
    graph = spark_dsg.DynamicSceneGraph.load(str(scene_path))
    objects, rooms = prepare_labels(graph, object_labelspace, room_labelspace)
    summarize_dsg(graph)
    with tempfile.TemporaryDirectory(prefix="heracles-benchmark-labels-") as directory:
        paths = []
        for name, labels in (("objects", objects), ("rooms", rooms)):
            path = Path(directory) / f"{name}.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "label_names": [
                            {"label": label, "name": value}
                            for label, value in sorted(labels.items())
                        ]
                    }
                )
            )
            paths.append(str(path))
        load_dsg_to_db(paths[0], paths[1], uri, credentials, graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_graph", type=Path, required=True)
    parser.add_argument("--neo4j_uri", default=os.getenv("HERACLES_NEO4J_URI"))
    parser.add_argument("--object_labelspace")
    parser.add_argument("--room_labelspace")
    args = parser.parse_args()
    credentials = (os.getenv("HERACLES_NEO4J_USERNAME"), os.getenv("HERACLES_NEO4J_PASSWORD"))
    if not args.neo4j_uri or not all(credentials):
        parser.error("Set HERACLES_NEO4J_URI, HERACLES_NEO4J_USERNAME, and HERACLES_NEO4J_PASSWORD")
    load_scene(
        args.scene_graph,
        args.neo4j_uri,
        credentials,
        args.object_labelspace,
        args.room_labelspace,
    )


if __name__ == "__main__":
    main()

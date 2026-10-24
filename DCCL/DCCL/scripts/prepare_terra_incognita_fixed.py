#!/usr/bin/env python3
"""Rebuild DomainBed TerraIncognita from existing class-organized images.

Uses image["location"] and category annotations from all five official JSONs.
Follows the locations/classes in facebookresearch/DomainBed's download script:
https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py
The new dataset contains image symlinks; original images and directories are
left intact. Validation completes before the output directory is created.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile


LOCATIONS = ("100", "38", "43", "46")
CLASSES = (
    "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum",
    "rabbit", "raccoon", "squirrel",
)
ANNOTATION_FILES = (
    "cis_test_annotations.json", "cis_val_annotations.json",
    "train_annotations.json", "trans_test_annotations.json",
    "trans_val_annotations.json",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm",
                    ".tif", ".tiff", ".webp"}
FORMAT = "domainbed-terra-four-domain-symlinks-v1"


def load_annotations(annotation_root):
    images = {}
    category_names = {}
    labels = defaultdict(set)
    hashes = {}
    for filename in ANNOTATION_FILES:
        path = annotation_root / filename
        if not path.is_file():
            raise ValueError("Annotation file not found: {}".format(path))
        payload = path.read_bytes()
        hashes[filename] = hashlib.sha256(payload).hexdigest()
        data = json.loads(payload)
        for category in data["categories"]:
            key, name = str(category["id"]), category["name"]
            if key in category_names and category_names[key] != name:
                raise ValueError("Conflicting category ID {}".format(key))
            category_names[key] = name
        for image in data["images"]:
            key = str(image["id"])
            value = (str(image["location"]), image["file_name"])
            if key in images and images[key] != value:
                raise ValueError("Conflicting image metadata for {}".format(key))
            images[key] = value
        for annotation in data["annotations"]:
            labels[str(annotation["image_id"])].add(str(annotation["category_id"]))
    return images, category_names, labels, hashes


def build_plan(images_root, images, category_names, labels):
    if not images_root.is_dir():
        raise ValueError("Image directory not found: {}".format(images_root))
    by_name = defaultdict(list)
    for path in images_root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            by_name[path.name.casefold()].append(path)
    print("Indexed {} existing image files.".format(
        sum(len(paths) for paths in by_name.values())), flush=True)

    plan = {}
    unresolved = []
    for image_id, (location, filename) in sorted(images.items()):
        if location not in LOCATIONS:
            continue
        relative = PurePosixPath(filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe annotation filename: {}".format(filename))
        for category_id in sorted(labels.get(image_id, ())):
            if category_id not in category_names:
                raise ValueError("Unknown category ID {}".format(category_id))
            category = category_names[category_id]
            if category not in CLASSES:
                continue
            source = None
            for candidate in (
                images_root.joinpath(*relative.parts),
                images_root / category / relative.name,
            ):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                    source = candidate.resolve()
                    break
            if source is None:
                candidates = by_name.get(relative.name.casefold(), [])
                if len(candidates) == 1:
                    source = candidates[0].resolve()
                else:
                    in_class = [
                        path for path in candidates
                        if path.relative_to(images_root).parts[0] == category
                    ]
                    if len(in_class) == 1:
                        source = in_class[0].resolve()
            if source is None:
                unresolved.append("{} (class={}, location={}, candidates={})".format(
                    filename, category, location,
                    len(by_name.get(relative.name.casefold(), []))))
                continue
            destination = Path("L" + location) / category / relative.name
            if destination in plan and plan[destination] != source:
                raise ValueError("Output filename collision: {}".format(destination))
            # The official code copies once per annotation. Repeated copies
            # overwrite the same path; one link gives the identical sample set.
            # Distinct category assignments retain distinct class-folder entries.
            plan[destination] = source

    if unresolved:
        raise ValueError(
            "{} selected image/category entries could not be resolved. "
            "No output was created.\n{}".format(
                len(unresolved), "\n".join(unresolved[:12])))
    counts = Counter((path.parts[0], path.parts[1]) for path in plan)
    empty = [
        "L{}/{}".format(location, category)
        for location in LOCATIONS for category in CLASSES
        if not counts[("L" + location, category)]
    ]
    if empty:
        raise ValueError(
            "Missing domain/class groups; dataset was not created: " + ", ".join(empty))
    return plan, counts


def validate_existing(output_root, plan):
    manifest = output_root / "preparation.json"
    if output_root.is_symlink() or not output_root.is_dir() or not manifest.is_file():
        raise ValueError("Output already exists; it was not modified: {}".format(output_root))
    if json.loads(manifest.read_text())["format"] != FORMAT:
        raise ValueError("Unrecognized existing dataset; it was not modified.")
    domains = {p.name for p in output_root.iterdir() if p.is_dir()}
    if domains != {"L" + location for location in LOCATIONS}:
        raise ValueError("Existing output has an unexpected domain set.")
    for domain in domains:
        actual_classes = {p.name for p in (output_root / domain).iterdir() if p.is_dir()}
        if actual_classes != set(CLASSES):
            raise ValueError("Existing output has an unexpected class set.")
    for relative, source in plan.items():
        link = output_root / relative
        if not link.is_symlink() or not link.is_file() or link.resolve() != source:
            raise ValueError("Existing output differs from the source: {}".format(link))
    actual = {
        p.relative_to(output_root) for p in output_root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and (p.is_file() or p.is_symlink())
    }
    if actual != set(plan):
        raise ValueError("Existing output contains missing or extra image entries.")


def prepare(images_root, annotation_root, output_root):
    images_root = images_root.resolve()
    annotation_root = annotation_root.resolve()
    output_root = output_root.absolute()
    images, categories, labels, hashes = load_annotations(annotation_root)
    plan, counts = build_plan(images_root, images, categories, labels)
    if os.path.lexists(output_root):
        validate_existing(output_root, plan)
        print("Existing dataset verified; no changes needed.")
    else:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(
            prefix=output_root.name + "_building_", dir=output_root.parent))
        try:
            for relative, source in sorted(plan.items()):
                link = stage / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source)
            manifest = {
                "format": FORMAT,
                "images_root": str(images_root),
                "annotation_root": str(annotation_root),
                "annotation_sha256": hashes,
                "locations": list(LOCATIONS),
                "classes": list(CLASSES),
                "image_entries": len(plan),
                "domain_class_counts": {
                    domain + "/" + category: count
                    for (domain, category), count in sorted(counts.items())
                },
            }
            (stage / "preparation.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            # Recheck each image link before publishing the completed directory.
            if any(not (stage / relative).is_file() for relative in plan):
                raise ValueError("An image link became unavailable during preparation.")
            if os.path.lexists(output_root):
                raise ValueError("Output appeared during preparation; refusing to replace it.")
            stage.rename(output_root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    for location in LOCATIONS:
        total = sum(counts[("L" + location, category)] for category in CLASSES)
        print("L{}: {} image entries, {} classes".format(location, total, len(CLASSES)))
    print("READY: {}".format(output_root))
    print("Use --data_dir {}".format(output_root.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Parent directory containing the existing Terra data.")
    parser.add_argument("--images-root", type=Path,
                        help="Override the existing class-organized image directory.")
    parser.add_argument("--annotations-dir", type=Path,
                        help="Override the five official annotation JSONs' directory.")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing datasets are never overwritten.")
    args = parser.parse_args()
    images_root = args.images_root or args.data_dir / "terra_incognita0/terra_incognita_raw"
    annotation_root = args.annotations_dir or args.data_dir / "eccv_18_annotation_files"
    output_root = args.output_dir or args.data_dir / "terra_incognita_fixed"
    try:
        prepare(images_root, annotation_root, output_root)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()

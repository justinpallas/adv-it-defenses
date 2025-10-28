"""Helpers for working with ImageNet label metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from scipy.io import loadmat
from torchvision.models import ResNet50_Weights


def load_ground_truth_labels(ground_truth_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return filename-to-index and label name-to-index mappings for ImageNet."""
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file {ground_truth_path} not found.")

    meta_path = ground_truth_path.parent / "meta.mat"
    ilsvrc_id_to_wnid, wnid_to_words = load_synset_metadata(meta_path)
    categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    wnid_to_index = build_wnid_to_class_index(wnid_to_words, categories)
    label_to_index = build_label_to_index(wnid_to_words, wnid_to_index)

    labels: Dict[str, int] = {}
    with ground_truth_path.open("r") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            ilsvrc_id = int(line)
            wnid = ilsvrc_id_to_wnid.get(ilsvrc_id)
            if wnid is None:
                raise KeyError(f"No WNID found for ILSVRC2012 ID {ilsvrc_id}.")
            mapped_index = wnid_to_index.get(wnid)
            if mapped_index is None:
                raise KeyError(
                    f"No class index found for WNID '{wnid}'. Ensure torchvision is available."
                )

            base_name = f"ILSVRC2012_val_{idx:08d}"
            labels[f"{base_name}.JPEG"] = mapped_index
            labels[f"{base_name}.jpg"] = mapped_index
            labels[f"{base_name}.png"] = mapped_index
            labels[f"{base_name}.PNG"] = mapped_index
    if not labels:
        raise ValueError(f"No labels parsed from {ground_truth_path}.")
    return labels, label_to_index


def load_synset_metadata(meta_path: Path) -> Tuple[Dict[int, str], Dict[str, str]]:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.mat not found at {meta_path}.")

    meta_raw = loadmat(meta_path, squeeze_me=True, struct_as_record=False)["synsets"]
    leaves = [entry for entry in meta_raw if int(entry.num_children) == 0]

    id_to_wnid: Dict[int, str] = {}
    wnid_to_words: Dict[str, str] = {}
    for entry in leaves:
        wnid = str(entry.WNID)
        id_to_wnid[int(entry.ILSVRC2012_ID)] = wnid
        wnid_to_words[wnid] = str(entry.words)

    if not id_to_wnid:
        raise ValueError("Failed to extract any synset mappings from meta.mat.")
    return id_to_wnid, wnid_to_words


def build_wnid_to_class_index(wnid_to_words: Dict[str, str], categories: Sequence[str]) -> Dict[str, int]:
    wnid_to_idx: Dict[str, int] = {}
    missing: list[str] = []
    overrides = {
        "crane": "n03126707",
        "crane bird": "n02012849",
    }

    synset_infos = []
    info_by_wnid: Dict[str, dict] = {}
    for wnid, words in wnid_to_words.items():
        synonyms = [syn.strip() for syn in words.split(",")]
        lower_synonyms = [syn.lower() for syn in synonyms]
        combined = " ".join(synonyms).replace("  ", " ").strip()
        info = {
            "synonyms": synonyms,
            "lower_synonyms": lower_synonyms,
            "combined": combined,
            "lower_combined": combined.lower(),
        }
        synset_infos.append((wnid, info))
        info_by_wnid[wnid] = info

    for idx, category in enumerate(categories):
        if category in overrides:
            wnid_to_idx[overrides[category]] = idx
            continue

        lower_category = category.lower()

        def match(predicate) -> list[str]:
            return [wnid for wnid, info in synset_infos if predicate(info)]

        matches = match(lambda info: category == info["synonyms"][0])
        if not matches:
            matches = match(lambda info: category in info["synonyms"])
        if not matches:
            matches = match(lambda info: category == info["combined"])
        if not matches:
            matches = match(lambda info: info["synonyms"][0].startswith(category))
        if not matches:
            matches = match(lambda info: lower_category == info["synonyms"][0].lower())
        if not matches:
            matches = match(lambda info: lower_category in info["lower_synonyms"])
        if not matches:
            matches = match(lambda info: lower_category == info["lower_combined"])
        if not matches:
            missing.append(category)
            continue

        if len(matches) > 1:
            matches = sorted(matches, key=lambda wnid: info_by_wnid[wnid]["combined"])
        wnid_to_idx[matches[0]] = idx

    if missing:
        raise ValueError(f"Failed to map categories to WNIDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return wnid_to_idx


def build_label_to_index(wnid_to_words: Dict[str, str], wnid_to_idx: Dict[str, int]) -> Dict[str, int]:
    label_to_idx: Dict[str, int] = {}

    for wnid, index in wnid_to_idx.items():
        words = wnid_to_words[wnid]
        synonyms = [syn.strip() for syn in words.split(",")]

        variants: set[str] = set()
        variants.add(words.strip())
        variants.add(", ".join(synonyms))
        variants.add(" ".join(synonyms))
        variants.update(synonyms)

        lower_variants = {variant.lower() for variant in variants}
        variants.update(lower_variants)

        for variant in variants:
            label_to_idx.setdefault(variant, index)

    return label_to_idx

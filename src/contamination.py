"""Единая проверка пересечений для split и отдельного CI-гейта."""

from typing import Sequence

from src.dedup import cross_near_duplicates
from src.schema import Example
from src.textnorm import normalize_group, normalize_text


def report(
    train: Sequence[Example],
    test: Sequence[Example],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> dict:
    train_ids = {example.id for example in train}
    test_ids = {example.id for example in test}
    train_texts = [normalize_text(example.user) for example in train]
    test_texts = [normalize_text(example.user) for example in test]
    train_groups = {normalize_group(example.topic) for example in train}
    test_groups = {normalize_group(example.topic) for example in test}
    id_overlap = sorted(train_ids & test_ids)
    text_overlap = sorted(set(train_texts) & set(test_texts))
    group_overlap = sorted(train_groups & test_groups)
    pairs = cross_near_duplicates(
        train_texts, test_texts, shingle_words, num_perm, threshold
    )
    return {
        "id_overlap": len(id_overlap),
        "text_overlap": len(text_overlap),
        "group_overlap": len(group_overlap),
        "near_dup_pairs": len(pairs),
        "examples": {
            "id": id_overlap[:3],
            "group": group_overlap[:3],
            "near_dup": [
                {"train": train[left].id, "test": test[right].id}
                for left, right in pairs[:3]
            ],
        },
    }


def is_clean(result: dict) -> bool:
    return all(result[key] == 0 for key in ("id_overlap", "text_overlap", "group_overlap", "near_dup_pairs"))

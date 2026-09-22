"""Точная и приближённая дедупликация по словным шинглам."""

from typing import Sequence

from datasketch import MinHash, MinHashLSH

from src.textnorm import shingles


def build_minhash(text: str, shingle_words: int, num_perm: int) -> MinHash:
    result = MinHash(num_perm=num_perm)
    result.update_batch([item.encode("utf-8") for item in shingles(text, shingle_words)])
    return result


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    seen: set[str] = set()
    duplicates: list[int] = []
    for index, key in enumerate(keys):
        if key in seen:
            duplicates.append(index)
        else:
            seen.add(key)
    return duplicates


def near_duplicates(
    texts: Sequence[str], shingle_words: int, num_perm: int, threshold: float
) -> list[int]:
    """Жадно оставить первого представителя каждого near-duplicate кластера."""
    index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicates: list[int] = []
    for position, text in enumerate(texts):
        signature = build_minhash(text, shingle_words, num_perm)
        if index.query(signature):
            duplicates.append(position)
        else:
            index.insert(str(position), signature)
    return duplicates


def cross_near_duplicates(
    left: Sequence[str],
    right: Sequence[str],
    shingle_words: int,
    num_perm: int,
    threshold: float,
) -> list[tuple[int, int]]:
    index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for position, text in enumerate(left):
        index.insert(str(position), build_minhash(text, shingle_words, num_perm))
    pairs: list[tuple[int, int]] = []
    for right_position, text in enumerate(right):
        signature = build_minhash(text, shingle_words, num_perm)
        pairs.extend((int(key), right_position) for key in index.query(signature))
    return pairs

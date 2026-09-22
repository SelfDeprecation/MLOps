#!/usr/bin/env python3
"""Отдельный исполняемый гейт контаминации train/test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import iter_examples


def main() -> int:
    params = load_params()
    paths = params["paths"]
    near_cfg = params["clean"]["near_dup"]
    train = list(iter_examples(paths["train"]))
    test = list(iter_examples(paths["test"]))
    result = report(
        train,
        test,
        near_cfg["shingle_words"],
        near_cfg["num_perm"],
        params["contamination"]["threshold"],
    )
    print(f"train: {len(train)} строк, test: {len(test)} строк")
    print(f"  пересечение по id:        {result['id_overlap']}")
    print(f"  пересечение по тексту:    {result['text_overlap']}")
    print(f"  пересечение по группам:   {result['group_overlap']}")
    print(f"  near-dup пар train/test:  {result['near_dup_pairs']}")
    if is_clean(result):
        print("контаминации нет")
        return 0
    print(f"КОНТАМИНАЦИЯ: {result['examples']}")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

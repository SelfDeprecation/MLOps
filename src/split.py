"""Групповой детерминированный split без контаминации тем."""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def assign_groups(sizes: dict[str, int], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Распределить целые группы, минимизируя недобор целевого числа строк."""
    if not sizes:
        return {}
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or any(value <= 0 for value in ratios.values()):
        raise ValueError("split.ratios должны быть положительными и суммироваться в 1")
    groups = list(sizes)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda key: sizes[key], reverse=True)
    targets = {name: sum(sizes.values()) * ratio for name, ratio in ratios.items()}
    assigned = {name: 0 for name in ratios}
    result: dict[str, str] = {}
    for group in groups:
        destination = max(ratios, key=lambda name: (targets[name] - assigned[name], -assigned[name]))
        result[group] = destination
        assigned[destination] += sizes[group]
    return result


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")
    started = time.perf_counter()
    examples: list[Example] = list(iter_examples(paths["clean"]))
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[normalize_group(example.topic)].append(example)
    labels = assign_groups({key: len(value) for key, value in grouped.items()}, cfg["ratios"], cfg["seed"])
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for group, rows in grouped.items():
        buckets[labels[group]].extend(rows)

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for example in rows:
                fh.write(dump(example) + "\n")

    near_cfg = params["clean"]["near_dup"]
    contamination = report(
        buckets["train"],
        buckets["test"],
        near_cfg["shingle_words"],
        near_cfg["num_perm"],
        params["contamination"]["threshold"],
    )
    if not is_clean(contamination):
        raise RuntimeError(f"контаминация после split: {contamination}")

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(grouped),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(example.topic) for example in rows})
            for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": contamination,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("split: " + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items()))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()

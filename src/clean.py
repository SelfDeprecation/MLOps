"""Стадия clean: схема, длины, PII, точные и приближённые дубли."""

import json
import time
from pathlib import Path

from src.config import load_params
from src.dedup import exact_duplicates, near_duplicates
from src.pii import scrub
from src.schema import Example, dump, iter_examples
from src.stats import percentile
from src.textnorm import normalize_group, normalize_text


def percentiles(values: list[int]) -> dict[str, int]:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
    }


def main() -> None:
    params = load_params()
    cfg = params["clean"]
    paths = params["paths"]
    started = time.perf_counter()
    examples: list[Example] = list(iter_examples(paths["raw"]))
    rows_in = len(examples)

    kept: list[Example] = []
    dropped_length = 0
    for example in examples:
        if not (cfg["min_user_chars"] <= len(example.user) <= cfg["max_user_chars"]):
            dropped_length += 1
        elif len(example.assistant) < cfg["min_assistant_chars"]:
            dropped_length += 1
        else:
            kept.append(example)

    pii_hits: dict[str, int] = {}
    pii_rows = 0
    if cfg["pii"]["enabled"]:
        for example in kept:
            touched = False
            for message in example.messages:
                message.content, hits = scrub(message.content)
                touched = touched or bool(hits)
                for name, count in hits.items():
                    pii_hits[name] = pii_hits.get(name, 0) + count
            pii_rows += int(touched)

    keys = [normalize_text(example.user) for example in kept]
    exact = set(exact_duplicates(keys))
    kept = [example for index, example in enumerate(kept) if index not in exact]

    near: set[int] = set()
    near_cfg = cfg["near_dup"]
    if near_cfg["enabled"]:
        near = set(
            near_duplicates(
                [normalize_text(example.user) for example in kept],
                near_cfg["shingle_words"],
                near_cfg["num_perm"],
                near_cfg["threshold"],
            )
        )
        kept = [example for index, example in enumerate(kept) if index not in near]

    out = Path(paths["clean"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for example in kept:
            fh.write(dump(example) + "\n")

    metrics = {
        "version": params["collect"]["version"],
        "rows_in": rows_in,
        "rows_out": len(kept),
        "dropped_length": dropped_length,
        "dropped_exact_dup": len(exact),
        "dropped_near_dup": len(near),
        "pii_rows_masked": pii_rows,
        "pii_hits": {name: pii_hits.get(name, 0) for name in ("phone", "email", "birth_date")},
        "groups": len({normalize_group(example.topic) for example in kept}),
        "user_chars": percentiles([len(example.user) for example in kept]),
        "assistant_chars": percentiles([len(example.assistant) for example in kept]),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_clean"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"clean: {rows_in} -> {len(kept)} строк (длина -{dropped_length}, "
        f"точные -{len(exact)}, near-dup -{len(near)}), ПДн в {pii_rows} строках"
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()

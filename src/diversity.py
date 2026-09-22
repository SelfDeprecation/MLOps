"""Машинный гейт разнообразия очищенного набора."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.config import load_params
from src.schema import iter_examples
from src.stats import spread
from src.textnorm import normalize_group, normalize_text


class DiversityError(ValueError):
    """Датасет формально валиден, но нарушает пороги разнообразия."""


def measure(path: str, group_key: str) -> dict:
    examples = list(iter_examples(path))
    if not examples:
        raise DiversityError(f"{path}: ни одной строки")
    if group_key != "topic":
        raise DiversityError(f"неподдерживаемый group_key: {group_key}")
    systems = {normalize_text(example.messages[0].content) for example in examples}
    groups = Counter(normalize_group(example.topic) for example in examples)
    answers = [example.assistant for example in examples]
    lengths = [len(answer) for answer in answers]
    length_counts = Counter(lengths)
    answer_counts = Counter(normalize_text(answer) for answer in answers)
    largest_group, largest_group_count = groups.most_common(1)[0]
    _, top_length_count = length_counts.most_common(1)[0]
    duplicate_answers = sum(count - 1 for count in answer_counts.values() if count > 1)
    return {
        "examples": len(examples),
        "system_prompts": len(systems),
        "groups": len(groups),
        "largest_group": largest_group,
        "largest_group_share": round(largest_group_count / len(examples), 4),
        "answer_len": spread(lengths),
        "same_length_share": round(top_length_count / len(examples), 4),
        "duplicate_answer_share": round(duplicate_answers / len(examples), 4),
    }


def violations(stats: dict, cfg: dict) -> list[str]:
    found: list[str] = []
    if stats["examples"] < cfg["min_examples"]:
        found.append(f"мало примеров: {stats['examples']}, нужно ≥ {cfg['min_examples']}")
    if stats["system_prompts"] < cfg["min_system_prompts"]:
        found.append(
            f"системных промптов {stats['system_prompts']}, нужно ≥ {cfg['min_system_prompts']}"
        )
    if stats["groups"] < cfg["min_groups"]:
        found.append(f"групп {stats['groups']}, нужно ≥ {cfg['min_groups']}")
    if stats["largest_group_share"] > cfg["max_group_share"]:
        found.append(
            f"крупнейшая группа занимает {stats['largest_group_share']:.1%} "
            f"(порог {cfg['max_group_share']:.0%})"
        )
    ratio = stats["answer_len"]["ratio_p90_p10"]
    if ratio < cfg["min_answer_len_ratio"]:
        found.append(f"разброс длин ответа p90/p10 = {ratio}, нужно ≥ {cfg['min_answer_len_ratio']}")
    if stats["same_length_share"] > cfg["max_same_length_share"]:
        found.append(
            f"{stats['same_length_share']:.1%} ответов имеют одну и ту же длину "
            f"(порог {cfg['max_same_length_share']:.0%})"
        )
    if stats["duplicate_answer_share"] > cfg["max_duplicate_answer_share"]:
        found.append(
            f"{stats['duplicate_answer_share']:.1%} ответов дословно повторяются "
            f"(порог {cfg['max_duplicate_answer_share']:.0%})"
        )
    return found


def main() -> None:
    params = load_params()
    cfg = params["diversity"]
    paths = params["paths"]
    started = time.perf_counter()
    stats = measure(paths["clean"], params["split"]["group_key"])
    failed = violations(stats, cfg)
    metrics = {
        "version": params["collect"]["version"],
        **stats,
        "thresholds": dict(cfg),
        "violations": failed,
        "passed": not failed,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_diversity"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        print("diversity: нарушены пороги:", file=sys.stderr)
        for item in failed:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"diversity: {stats['examples']} строк, {stats['system_prompts']} системных промптов, "
        f"{stats['groups']} групп, p90/p10={stats['answer_len']['ratio_p90_p10']}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

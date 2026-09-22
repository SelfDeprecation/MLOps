"""Стадия collect: авторский каталог MLOps → chat JSONL."""

import hashlib
import json
import time
from pathlib import Path

from src.config import load_params
from src.dataset_catalog import QUESTION_PATTERNS, TOPICS, Topic


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать системную инструкцию по id."""
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def make_answer(topic: Topic, index: int) -> str:
    """Сформировать ответы разной длины без случайности и скрытого состояния."""
    lead = f"Наблюдение: {topic.signal}."
    suffix = (
        f" Контрольный замер: окно {12 + index * 3} минут, "
        f"не менее {40 + index * 7} наблюдений."
    )
    if index % 4 == 0:
        return f"{lead} Первый шаг — {topic.action}. Результат нужно сохранить как проверяемый артефакт.{suffix}"
    if index % 4 == 1:
        return (
            f"{lead} Это согласуется с понятием «{topic.name}»: {topic.meaning}. "
            f"Сначала следует {topic.action}, затем повторить измерение на том же срезе "
            f"и сравнить его с заранее зафиксированным baseline.{suffix}"
        )
    if index % 4 == 2:
        return (
            f"Факт пока только один: {topic.signal}. Сам по себе он не доказывает причину. "
            f"Для проверки гипотезы «{topic.name}» нужно {topic.action}. "
            "В журнале эксперимента стоит сохранить входной срез, версию кода, время, "
            f"полученную метрику и порог решения; тогда вывод можно воспроизвести.{suffix}"
        )
    return (
        f"«{topic.name}» означает, что {topic.meaning}. Наблюдаемый симптом — {topic.signal} — "
        "нужно сначала локализовать по среде и сегменту, не объявляя его причиной. "
        f"Безопасный первый шаг: {topic.action}. После этого сравнивают результат с baseline, "
        "проверяют соседние стадии и только затем выбирают исправление. Критерий успеха и "
        f"условие отката фиксируются до изменения production.{suffix}"
    )


def make_record(topic: Topic, topic_index: int, example_index: int, prompts: list[str]) -> dict:
    example_id = f"mlops_{topic_index:02d}_{example_index:02d}"
    pattern = QUESTION_PATTERNS[example_index % len(QUESTION_PATTERNS)]
    question = pattern.format(
        name=topic.name,
        meaning=topic.meaning,
        signal=topic.signal,
        minutes=7 + example_index * 3,
        environment=("production", "staging", "контуре обучения")[example_index % 3],
        case=100 + topic_index * 24 + example_index,
        budget=1 + example_index % 5,
        factor=2 + example_index % 4,
        sample=40 + example_index * 7,
    )
    return {
        "id": example_id,
        "topic": topic.name,
        "messages": [
            {"role": "system", "content": pick_prompt(example_id, prompts)},
            {"role": "user", "content": question},
            {"role": "assistant", "content": make_answer(topic, example_index)},
        ],
    }


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    version = cfg["version"]
    if version not in cfg["sources"]:
        raise SystemExit(f"неизвестная версия collect.version: {version!r}")
    prompts = list(cfg["system_prompts"])
    if not prompts:
        raise SystemExit("collect.system_prompts пуст")

    group_count = int(cfg["sources"][version]["groups"])
    per_topic = int(cfg["examples_per_topic"])
    if group_count > len(TOPICS):
        raise SystemExit(f"запрошено {group_count} тем, в каталоге только {len(TOPICS)}")

    started = time.perf_counter()
    rows = [
        make_record(topic, topic_index, example_index, prompts)
        for topic_index, topic in enumerate(TOPICS[:group_count])
        for example_index in range(per_topic)
    ]

    # Контролируемые шумовые случаи доказывают, что clean действительно работает.
    duplicate = json.loads(json.dumps(rows[0], ensure_ascii=False))
    duplicate["id"] = "quality_exact_duplicate"
    near_duplicate = json.loads(json.dumps(rows[1], ensure_ascii=False))
    near_duplicate["id"] = "quality_near_duplicate"
    near_duplicate["messages"][1]["content"] += " Повторите."
    pii_case = json.loads(json.dumps(rows[2], ensure_ascii=False))
    pii_case["id"] = "quality_pii_case"
    pii_case["messages"][1]["content"] += (
        " Контакт дежурного: +7 (999) 123-45-67, ops@example.ru; "
        "дата рождения 12.03.1975 указана в ошибочной выгрузке."
    )
    rows.extend([duplicate, near_duplicate, pii_case])

    out = Path(params["paths"]["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "version": version,
        "source": "author_catalog",
        "catalog_topics_available": len(TOPICS),
        "topics_selected": group_count,
        "examples_per_topic": per_topic,
        "base_rows": group_count * per_topic,
        "quality_cases_added": 3,
        "rows_written": len(rows),
        "system_prompt_variants": len({r["messages"][0]["content"] for r in rows}),
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(params["paths"]["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"collect: {version}, тем {group_count}, базовых строк {metrics['base_rows']}, "
        f"всего {len(rows)} -> {out}"
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()

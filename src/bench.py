"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пиковая RSS            — максимум за процесс, а не снимок в конце.
"""

import json
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch

from src.config import load_params
from src.model import generate, load_model, set_seed


class PeakRSSMonitor:
    """Фоновый кроссплатформенный замер максимальной RSS процесса."""

    def __init__(self, interval_sec: float = 0.01) -> None:
        self._process = psutil.Process()
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_bytes = self._process.memory_info().rss

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_sec):
            self.peak_bytes = max(
                self.peak_bytes,
                self._process.memory_info().rss,
            )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
        self._stop.set()
        self._thread.join()
        return self.peak_bytes / (1024 ** 2)


def synchronize(model) -> None:
    """Дождаться асинхронных вычислений ускорителя перед остановкой таймера."""
    device_type = model.device.type
    if device_type == "cuda":
        torch.cuda.synchronize(model.device)
    elif device_type == "mps":
        torch.mps.synchronize()


def main() -> None:
    params = load_params()
    prompt = params["bench"]["prompt"]

    rss_monitor = PeakRSSMonitor()
    rss_monitor.start()
    try:
        load_started = time.perf_counter()
        tokenizer, model = load_model(params)
        synchronize(model)
        load_time = time.perf_counter() - load_started

        for _ in range(params["bench"]["warmup_runs"]):
            set_seed(params["generate"]["seed"])
            generate(tokenizer, model, params, prompt)
            synchronize(model)

        speeds = []
        for _ in range(params["bench"]["measure_runs"]):
            set_seed(params["generate"]["seed"])
            synchronize(model)
            generation_started = time.perf_counter()
            _, n_tokens = generate(tokenizer, model, params, prompt)
            synchronize(model)
            elapsed = time.perf_counter() - generation_started
            speeds.append(n_tokens / elapsed)
    finally:
        peak_rss = rss_monitor.stop()

    # Медиана устойчивее среднего к одиночному выбросу.
    report = {
        "model": params["model"]["name"],
        "device": str(model.device),
        "dtype": params["model"]["dtype"],
        "load_time_sec": round(load_time, 2),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(s, 2) for s in speeds],
        "peak_rss_mb": round(peak_rss, 1),
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

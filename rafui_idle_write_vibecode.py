"""Idle baseline collection module for RAFUI."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from rafui_basic_print_vibecode import RafuiBLEClient, configure_logging

DEFAULT_IDLE_DURATION_S = 10.0
EXPECTED_SAMPLE_RATE_HZ = 50
HARDWARE_VERSION = "1.0"
SESSION_TYPE_IDLE = "idle"
JSON_DIR_NAME = "json"


def _iso_utc_timestamp() -> str:
	"""Return ISO-8601 UTC timestamp with Z suffix."""
	return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_filename_timestamp() -> str:
	"""Return filename-safe UTC timestamp."""
	return datetime.now(UTC).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _compute_baseline(samples: list[dict[str, float]]) -> dict[str, float]:
	"""Compute baseline summary statistics from idle samples."""
	if not samples:
		raise ValueError("No idle samples collected")

	values = np.array([[sample["vmag"], sample["vph"]] for sample in samples], dtype=np.float64)
	mean_vmag = float(np.mean(values[:, 0]))
	std_vmag = float(np.std(values[:, 0]))
	mean_vph = float(np.mean(values[:, 1]))
	std_vph = float(np.std(values[:, 1]))
	return {
		"mean_vmag": mean_vmag,
		"std_vmag": std_vmag,
		"mean_vph": mean_vph,
		"std_vph": std_vph,
	}


async def _collect_idle_samples_async(duration_s: float, client: RafuiBLEClient) -> list[dict[str, float]]:
	"""Collect idle samples asynchronously for the requested duration."""
	if duration_s <= 0:
		raise ValueError("duration_s must be > 0")

	samples: list[dict[str, float]] = []

	def on_sample(sample: dict[str, float]) -> None:
		samples.append(sample)

	stream_task = asyncio.create_task(client.stream(on_sample))

	with Progress(
		TextColumn("[bold cyan]Idle capture"),
		BarColumn(),
		TaskProgressColumn(),
		TimeElapsedColumn(),
	) as progress:
		task_id = progress.add_task("Collecting", total=duration_s)
		step_s = 0.1
		elapsed = 0.0
		while elapsed < duration_s:
			await asyncio.sleep(step_s)
			elapsed = min(elapsed + step_s, duration_s)
			progress.update(task_id, completed=elapsed)

	client.stop_stream()
	await stream_task
	await client.disconnect()
	return samples


def collect_idle_baseline(
	duration_s: float = DEFAULT_IDLE_DURATION_S,
	client: RafuiBLEClient | None = None,
	*,
	output_root: Path | None = None,
) -> dict[str, float]:
	"""Collect idle data, persist session JSON, and return baseline statistics.

	Args:
		duration_s: Capture duration in seconds.
		client: Optional pre-created BLE client.
		output_root: Optional project root containing the json folder.

	Returns:
		Dictionary containing idle baseline statistics.
	"""
	configure_logging()
	output_base = output_root if output_root is not None else Path(__file__).resolve().parent
	json_dir = output_base / JSON_DIR_NAME
	json_dir.mkdir(parents=True, exist_ok=True)

	active_client = client or RafuiBLEClient()

	try:
		samples = asyncio.run(_collect_idle_samples_async(duration_s=duration_s, client=active_client))
	except Exception as exc:
		raise RuntimeError(f"Idle collection failed: {exc}") from exc

	baseline = _compute_baseline(samples)

	payload: dict[str, Any] = {
		"session": SESSION_TYPE_IDLE,
		"timestamp": _iso_utc_timestamp(),
		"hardware_version": HARDWARE_VERSION,
		"sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
		"notes": "",
		"samples": samples,
		"baseline": baseline,
	}

	output_path = json_dir / f"idle_{_safe_filename_timestamp()}.json"
	try:
		output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	except OSError as exc:
		raise RuntimeError(f"Failed writing idle JSON: {exc}") from exc

	return baseline


if __name__ == "__main__":
	baseline_stats = collect_idle_baseline()
	print(json.dumps(baseline_stats, indent=2))


"""RAFUI model v2: period-based clustering with explicit per-period collection stats.

This module is designed for a calibration flow with explicit collection periods:
1) IDLE period
2) BTN_1 period
3) BTN_2 period
4) BTN_3 period

Each period is started and stopped manually by the operator. The module tracks:
- elapsed timer per period
- sample count per period
- raw samples per period

Modeling approach (cluster-based, different from v1):
- Build sliding-window features per period.
- Compute one robust medoid prototype per class from that class's own feature cloud.
- Classify new windows by nearest class medoid in scaled feature space.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 100
WINDOW_STEP = 1
NUM_FEATURES = 10

IDLE_LABEL = "IDLE"
BTN1_LABEL = "BTN_1"
BTN2_LABEL = "BTN_2"
BTN3_LABEL = "BTN_3"
NOISE_LABEL = "NOISE"

# Default class set includes idle plus 3 buttons.
DEFAULT_CLASS_LABELS: tuple[str, ...] = (IDLE_LABEL, BTN1_LABEL, BTN2_LABEL, BTN3_LABEL)

NOISE_DISTANCE_MULTIPLIER = 1.5
MIN_SAMPLES_PER_PERIOD = WINDOW_SIZE


@dataclass(slots=True)
class PeriodRecord:
    """One explicitly captured period with timing and sample stats.

    Attributes:
        label: Semantic label for the period (IDLE, BTN_1, BTN_2, BTN_3).
        started_at_iso: UTC timestamp when period started.
        ended_at_iso: UTC timestamp when period ended.
        elapsed_s: Duration in seconds.
        sample_count: Number of samples collected in this period.
        samples: Raw sample list, each item includes t, vmag, vph.
    """

    label: str
    started_at_iso: str
    ended_at_iso: str
    elapsed_s: float
    sample_count: int
    samples: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class RafuiModel2:
    """Period-based prototype clustering model."""

    class_labels: tuple[str, ...]
    scaler: StandardScaler
    class_medoids_scaled: np.ndarray
    class_medoids_unscaled: np.ndarray
    intra_class_distances: np.ndarray
    noise_distance_multiplier: float = NOISE_DISTANCE_MULTIPLIER


class PeriodCollector:
    """Collects manual start/stop periods and tracks timer + sample counts."""

    def __init__(self) -> None:
        self._active_label: str | None = None
        self._active_start_t_monotonic: float | None = None
        self._active_start_iso: str | None = None
        self._active_samples: list[dict[str, float]] = []
        self._records: list[PeriodRecord] = []

    @property
    def is_active(self) -> bool:
        """Return True if a period is currently collecting."""
        return self._active_label is not None

    @property
    def active_label(self) -> str | None:
        """Return currently active period label, if any."""
        return self._active_label

    @property
    def records(self) -> list[PeriodRecord]:
        """Return completed period records in capture order."""
        return list(self._records)

    def start_period(self, label: str) -> None:
        """Start a new collection period.

        Args:
            label: Period label, e.g. IDLE, BTN_1, BTN_2, BTN_3.

        Raises:
            RuntimeError: If another period is already active.
        """
        if self.is_active:
            raise RuntimeError(f"Period '{self._active_label}' is already active")

        self._active_label = label
        self._active_start_t_monotonic = _now_monotonic_s()
        self._active_start_iso = _iso_utc_timestamp()
        self._active_samples = []

    def add_sample(self, sample: dict[str, float]) -> None:
        """Append one sample to current active period.

        Sample must contain numeric t, vmag, vph keys.
        """
        if not self.is_active:
            return

        try:
            normalized = {
                "t": float(sample["t"]),
                "vmag": float(sample["vmag"]),
                "vph": float(sample["vph"]),
            }
        except (KeyError, TypeError, ValueError):
            return

        self._active_samples.append(normalized)

    def get_active_stats(self) -> dict[str, float | int | str | None]:
        """Return live stats for the active period (timer + sample count)."""
        if not self.is_active or self._active_start_t_monotonic is None:
            return {
                "label": None,
                "elapsed_s": 0.0,
                "sample_count": 0,
            }

        elapsed_s = max(0.0, _now_monotonic_s() - self._active_start_t_monotonic)
        return {
            "label": self._active_label,
            "elapsed_s": float(elapsed_s),
            "sample_count": int(len(self._active_samples)),
        }

    def end_period(self) -> PeriodRecord:
        """End current period and return finalized record."""
        if not self.is_active:
            raise RuntimeError("No active period to end")
        if self._active_start_t_monotonic is None or self._active_start_iso is None:
            raise RuntimeError("Active period has invalid timing state")

        elapsed_s = max(0.0, _now_monotonic_s() - self._active_start_t_monotonic)
        record = PeriodRecord(
            label=str(self._active_label),
            started_at_iso=self._active_start_iso,
            ended_at_iso=_iso_utc_timestamp(),
            elapsed_s=float(elapsed_s),
            sample_count=len(self._active_samples),
            samples=list(self._active_samples),
        )
        self._records.append(record)

        self._active_label = None
        self._active_start_t_monotonic = None
        self._active_start_iso = None
        self._active_samples = []

        return record

    def clear(self) -> None:
        """Reset collector state and drop all periods."""
        self._active_label = None
        self._active_start_t_monotonic = None
        self._active_start_iso = None
        self._active_samples = []
        self._records = []


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract 10 RAFUI features from one (100, 2) [vmag, vph] window."""
    if window.shape != (WINDOW_SIZE, 2):
        raise ValueError(f"window must be shape ({WINDOW_SIZE}, 2), got {window.shape}")

    vmag = window[:, 0]
    vph = window[:, 1]
    x = np.arange(WINDOW_SIZE, dtype=np.float64)

    mean_vmag = float(np.mean(vmag))
    std_vmag = float(np.std(vmag))
    mean_vph = float(np.mean(vph))
    std_vph = float(np.std(vph))

    slope_vmag = float(np.polyfit(x, vmag, 1)[0])
    slope_vph = float(np.polyfit(x, vph, 1)[0])

    delta_vmag_max = float(np.max(np.abs(np.diff(vmag))))
    delta_vph_max = float(np.max(np.abs(np.diff(vph))))

    zcr_vmag = _zero_crossing_rate(vmag - mean_vmag)
    zcr_vph = _zero_crossing_rate(vph - mean_vph)

    return np.array(
        [
            mean_vmag,
            std_vmag,
            mean_vph,
            std_vph,
            slope_vmag,
            slope_vph,
            delta_vmag_max,
            delta_vph_max,
            zcr_vmag,
            zcr_vph,
        ],
        dtype=np.float64,
    )


def train_from_periods(
    period_records: list[PeriodRecord],
    *,
    class_labels: tuple[str, ...] = DEFAULT_CLASS_LABELS,
) -> RafuiModel2:
    """Train v2 model from explicit period captures.

    Args:
        period_records: Completed records from PeriodCollector.
        class_labels: Class labels included in training order.

    Returns:
        Trained RafuiModel2.
    """
    if not period_records:
        raise ValueError("No period records provided")

    features_per_class: dict[str, np.ndarray] = {}
    for label in class_labels:
        samples = _samples_for_label(period_records, label)
        if len(samples) < MIN_SAMPLES_PER_PERIOD:
            raise ValueError(
                f"Label {label} has only {len(samples)} samples; need at least {MIN_SAMPLES_PER_PERIOD}"
            )
        raw = np.array([[s["vmag"], s["vph"]] for s in samples], dtype=np.float64)
        features_per_class[label] = _extract_sliding_feature_matrix(raw)

    concatenated = np.vstack([features_per_class[label] for label in class_labels])
    scaler = StandardScaler()
    concatenated_scaled = scaler.fit_transform(concatenated)

    # Split scaled features back by class.
    split_scaled: dict[str, np.ndarray] = {}
    cursor = 0
    for label in class_labels:
        n_rows = features_per_class[label].shape[0]
        split_scaled[label] = concatenated_scaled[cursor : cursor + n_rows]
        cursor += n_rows

    medoids_scaled: list[np.ndarray] = []
    medoids_unscaled: list[np.ndarray] = []
    intra_dist: list[float] = []

    for label in class_labels:
        points = split_scaled[label]
        medoid, mean_dist = _find_medoid_and_mean_distance(points)
        medoids_scaled.append(medoid)
        intra_dist.append(mean_dist)

        # For unscaled display/debugging.
        medoid_unscaled = scaler.inverse_transform(medoid.reshape(1, -1))[0]
        medoids_unscaled.append(medoid_unscaled)

    return RafuiModel2(
        class_labels=class_labels,
        scaler=scaler,
        class_medoids_scaled=np.vstack(medoids_scaled),
        class_medoids_unscaled=np.vstack(medoids_unscaled),
        intra_class_distances=np.array(intra_dist, dtype=np.float64),
    )


def train_from_period_file(
    data_path: str | Path,
    *,
    class_labels: tuple[str, ...] = DEFAULT_CLASS_LABELS,
) -> RafuiModel2:
    """Train v2 model from a saved period JSON file."""
    payload = _load_json(Path(data_path))
    records = _period_records_from_payload(payload)
    return train_from_periods(records, class_labels=class_labels)


def predict(window: np.ndarray, model: RafuiModel2) -> tuple[str, float]:
    """Predict class from one raw window using nearest class medoid.

    Returns:
        Tuple of (label, confidence in [0,1]).
    """
    features = extract_features(window).reshape(1, -1)
    scaled = model.scaler.transform(features)[0]

    deltas = model.class_medoids_scaled - scaled
    distances = np.linalg.norm(deltas, axis=1)
    nearest_idx = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_idx])

    base_distance = float(model.intra_class_distances[nearest_idx])
    if base_distance <= 1e-9:
        base_distance = 1e-9

    normalized = nearest_distance / base_distance
    confidence = max(0.0, min(1.0, 1.0 - (normalized / model.noise_distance_multiplier)))

    if normalized > model.noise_distance_multiplier:
        return (NOISE_LABEL, confidence)
    return (model.class_labels[nearest_idx], confidence)


def summarize_period_records(records: list[PeriodRecord]) -> list[dict[str, Any]]:
    """Return compact summary with timer and sample count per period."""
    return [
        {
            "label": r.label,
            "started_at": r.started_at_iso,
            "ended_at": r.ended_at_iso,
            "elapsed_s": r.elapsed_s,
            "sample_count": r.sample_count,
        }
        for r in records
    ]


def save_period_dataset(
    records: list[PeriodRecord],
    path: str | Path,
    *,
    hardware_version: str = "1.0",
    sample_rate_hz: int = 50,
    notes: str = "",
) -> Path:
    """Persist manual period captures with clear per-period stats."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "session": "training_periods_v2",
        "timestamp": _iso_utc_timestamp(),
        "hardware_version": hardware_version,
        "sample_rate_hz": sample_rate_hz,
        "notes": notes,
        "periods": [
            {
                "label": r.label,
                "started_at": r.started_at_iso,
                "ended_at": r.ended_at_iso,
                "elapsed_s": r.elapsed_s,
                "sample_count": r.sample_count,
                "samples": r.samples,
            }
            for r in records
        ],
        "period_summary": summarize_period_records(records),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_model(model: RafuiModel2, path: str | Path) -> Path:
    """Save model v2 to pickle."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)
    return output_path


def load_model(path: str | Path) -> RafuiModel2:
    """Load model v2 from pickle."""
    input_path = Path(path)
    with input_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, RafuiModel2):
        raise TypeError(f"Unexpected model type: {type(model)}")
    return model


def _samples_for_label(records: list[PeriodRecord], label: str) -> list[dict[str, float]]:
    all_samples: list[dict[str, float]] = []
    for record in records:
        if record.label == label:
            all_samples.extend(record.samples)
    return all_samples


def _extract_sliding_feature_matrix(values: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STEP):
        rows.append(extract_features(values[start : start + WINDOW_SIZE]))
    if not rows:
        raise ValueError("No feature windows could be extracted")
    return np.vstack(rows)


def _find_medoid_and_mean_distance(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Find geometric medoid among points and return mean distance to medoid."""
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be non-empty 2D array")

    # Pairwise distances matrix.
    diff = points[:, None, :] - points[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)

    medoid_idx = int(np.argmin(np.sum(dist_mat, axis=1)))
    medoid = points[medoid_idx]
    mean_dist = float(np.mean(dist_mat[medoid_idx]))
    mean_dist = max(mean_dist, 1e-9)
    return medoid, mean_dist


def _period_records_from_payload(payload: dict[str, Any]) -> list[PeriodRecord]:
    raw_periods = payload.get("periods")
    if not isinstance(raw_periods, list) or not raw_periods:
        raise ValueError("Expected non-empty 'periods' list in period dataset JSON")

    records: list[PeriodRecord] = []
    for item in raw_periods:
        label = str(item.get("label", ""))
        started = str(item.get("started_at", ""))
        ended = str(item.get("ended_at", ""))
        elapsed = float(item.get("elapsed_s", 0.0))
        samples = item.get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("Each period must include a samples list")

        normalized_samples: list[dict[str, float]] = []
        for s in samples:
            normalized_samples.append(
                {
                    "t": float(s["t"]),
                    "vmag": float(s["vmag"]),
                    "vph": float(s["vph"]),
                }
            )

        records.append(
            PeriodRecord(
                label=label,
                started_at_iso=started,
                ended_at_iso=ended,
                elapsed_s=elapsed,
                sample_count=len(normalized_samples),
                samples=normalized_samples,
            )
        )

    return records


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def _iso_utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_monotonic_s() -> float:
    # time.monotonic() is stable for period elapsed timing.
    import time

    return float(time.monotonic())

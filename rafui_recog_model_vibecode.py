"""Recognition model pipeline for RAFUI touch-state inference."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

WINDOW_SIZE = 100
WINDOW_STEP = 1
NUM_FEATURES = 10
NUM_CLUSTERS = 4
IDLE_LABEL = "IDLE"
BUTTON_LABELS = ("BTN_1", "BTN_2", "BTN_3")
NOISE_LABEL = "NOISE"
KMEDOIDS_METRIC = "euclidean"
KMEDOIDS_INIT = "k-medoids++"
RANDOM_SEED = 42
NOISE_DISTANCE_MULTIPLIER = 1.5
LOW_CONFIDENCE_FOR_TRANSITION = 0.4


@dataclass(slots=True)
class RafuiModel:
	"""Container for trained RAFUI clustering model artifacts."""

	medoid_centers_scaled: np.ndarray
	medoid_centers_unscaled: np.ndarray
	scaler: StandardScaler
	label_map: dict[int, str]
	intra_cluster_distances: np.ndarray
	noise_distance_multiplier: float = NOISE_DISTANCE_MULTIPLIER


def extract_features(window: np.ndarray) -> np.ndarray:
	"""Extract RAFUI features from a (100, 2) [vmag, vph] window.

	Args:
		window: Numpy array shaped (WINDOW_SIZE, 2).

	Returns:
		Numpy array of 10 engineered features.
	"""
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


def train(data_path: str, idle_baseline: dict[str, float]) -> RafuiModel:
	"""Train RAFUI model from captured training session JSON.

	Args:
		data_path: Path to training session JSON file.
		idle_baseline: Baseline dict from idle calibration module.

	Returns:
		Trained RafuiModel object.
	"""
	path = Path(data_path)
	if not path.exists():
		raise FileNotFoundError(f"Training file not found: {path}")

	payload = _load_json(path)
	samples = payload.get("samples", [])
	if len(samples) < WINDOW_SIZE:
		raise ValueError("Not enough samples for training window extraction")

	values = np.array([[float(s["vmag"]), float(s["vph"])] for s in samples], dtype=np.float64)
	feature_matrix = _extract_sliding_feature_matrix(values)

	scaler = StandardScaler()
	feature_matrix_scaled = scaler.fit_transform(feature_matrix)

	kmedoids = KMedoids(
		n_clusters=NUM_CLUSTERS,
		metric=KMEDOIDS_METRIC,
		init=KMEDOIDS_INIT,
		random_state=RANDOM_SEED,
	)
	cluster_indices = kmedoids.fit_predict(feature_matrix_scaled)

	medoid_scaled = np.asarray(kmedoids.cluster_centers_, dtype=np.float64)
	medoid_unscaled = scaler.inverse_transform(medoid_scaled)
	label_map = _assign_cluster_labels(medoid_unscaled, idle_baseline)
	intra_cluster = _mean_intra_cluster_distances(feature_matrix_scaled, cluster_indices, medoid_scaled)

	model = RafuiModel(
		medoid_centers_scaled=medoid_scaled,
		medoid_centers_unscaled=medoid_unscaled,
		scaler=scaler,
		label_map=label_map,
		intra_cluster_distances=intra_cluster,
	)

	_plot_training_clusters(feature_matrix, cluster_indices, medoid_unscaled, label_map)
	return model


def predict(window: np.ndarray, model: RafuiModel) -> tuple[str, float]:
	"""Predict state label and confidence from one raw window."""
	features = extract_features(window).reshape(1, -1)
	scaled = model.scaler.transform(features)[0]

	deltas = model.medoid_centers_scaled - scaled
	distances = np.linalg.norm(deltas, axis=1)
	nearest_idx = int(np.argmin(distances))
	nearest_distance = float(distances[nearest_idx])

	base_distance = float(model.intra_cluster_distances[nearest_idx])
	if base_distance <= 1e-9:
		base_distance = 1e-9
	normalized_distance = nearest_distance / base_distance

	confidence = max(
		0.0,
		min(1.0, 1.0 - (normalized_distance / model.noise_distance_multiplier)),
	)

	if normalized_distance > model.noise_distance_multiplier:
		return (NOISE_LABEL, confidence)

	label = model.label_map.get(nearest_idx, NOISE_LABEL)
	return (label, confidence)


def detect_transition(prev_state: str, curr_state: str, confidence: float) -> str | None:
	"""Detect state transition event from sequential inferred states.

	Returns:
		"ONSET", "OFFSET", or None.
	"""
	if confidence < LOW_CONFIDENCE_FOR_TRANSITION:
		return None

	prev_is_button = prev_state in BUTTON_LABELS
	curr_is_button = curr_state in BUTTON_LABELS

	if not prev_is_button and curr_is_button:
		return "ONSET"
	if prev_is_button and not curr_is_button and curr_state == IDLE_LABEL:
		return "OFFSET"
	return None


def save_model(model: RafuiModel, path: str | Path) -> None:
	"""Serialize model object to pickle file."""
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		with output_path.open("wb") as handle:
			pickle.dump(model, handle)
	except OSError as exc:
		raise RuntimeError(f"Failed to save model to {output_path}: {exc}") from exc


def load_model(path: str | Path) -> RafuiModel:
	"""Load RafuiModel from pickle file."""
	input_path = Path(path)
	try:
		with input_path.open("rb") as handle:
			loaded = pickle.load(handle)
	except OSError as exc:
		raise RuntimeError(f"Failed to load model from {input_path}: {exc}") from exc

	if not isinstance(loaded, RafuiModel):
		raise TypeError(f"Unexpected model type: {type(loaded)}")
	return loaded


def _extract_sliding_feature_matrix(values: np.ndarray) -> np.ndarray:
	"""Convert raw [vmag, vph] stream into feature matrix."""
	rows: list[np.ndarray] = []
	for start in range(0, len(values) - WINDOW_SIZE + 1, WINDOW_STEP):
		window = values[start : start + WINDOW_SIZE]
		rows.append(extract_features(window))
	if not rows:
		raise ValueError("No feature windows could be extracted")
	return np.vstack(rows)


def _assign_cluster_labels(
	medoid_unscaled: np.ndarray,
	idle_baseline: dict[str, float],
) -> dict[int, str]:
	"""Assign semantic labels to clusters based on medoid behavior."""
	if medoid_unscaled.shape[0] != NUM_CLUSTERS:
		raise ValueError("Expected 4 medoids for label assignment")

	# std features in vector are index 1 (vmag) and 3 (vph)
	idle_idx = int(np.argmin(medoid_unscaled[:, 1] + medoid_unscaled[:, 3]))

	idle_vmag = float(idle_baseline.get("mean_vmag", medoid_unscaled[idle_idx, 0]))
	remaining = [idx for idx in range(NUM_CLUSTERS) if idx != idle_idx]
	remaining.sort(key=lambda idx: abs(float(medoid_unscaled[idx, 0]) - idle_vmag))

	label_map: dict[int, str] = {idle_idx: IDLE_LABEL}
	for idx, button_label in zip(remaining, BUTTON_LABELS, strict=True):
		label_map[idx] = button_label
	return label_map


def _mean_intra_cluster_distances(
	features_scaled: np.ndarray,
	cluster_indices: np.ndarray,
	medoids_scaled: np.ndarray,
) -> np.ndarray:
	"""Compute per-cluster mean distance to medoid in scaled feature space."""
	means = np.zeros(NUM_CLUSTERS, dtype=np.float64)
	for cluster_idx in range(NUM_CLUSTERS):
		mask = cluster_indices == cluster_idx
		points = features_scaled[mask]
		if len(points) == 0:
			means[cluster_idx] = 1.0
			continue
		distances = np.linalg.norm(points - medoids_scaled[cluster_idx], axis=1)
		means[cluster_idx] = max(float(np.mean(distances)), 1e-9)
	return means


def _plot_training_clusters(
	feature_matrix: np.ndarray,
	cluster_indices: np.ndarray,
	medoid_unscaled: np.ndarray,
	label_map: dict[int, str],
) -> None:
	"""Plot mean_vmag vs mean_vph colored by assigned semantic labels."""
	plt.figure(figsize=(8, 6))
	for cluster_idx in range(NUM_CLUSTERS):
		mask = cluster_indices == cluster_idx
		label = label_map.get(cluster_idx, f"CLUSTER_{cluster_idx}")
		plt.scatter(
			feature_matrix[mask, 0],
			feature_matrix[mask, 2],
			s=12,
			alpha=0.6,
			label=label,
		)

	plt.scatter(
		medoid_unscaled[:, 0],
		medoid_unscaled[:, 2],
		s=240,
		marker="*",
		color="black",
		label="Medoids",
	)
	plt.title("RAFUI Training Clusters (mean_vmag vs mean_vph)")
	plt.xlabel("mean_vmag")
	plt.ylabel("mean_vph")
	plt.legend()
	plt.tight_layout()
	plt.show(block=False)


def _zero_crossing_rate(signal: np.ndarray) -> float:
	"""Compute zero-crossing rate around zero for a centered 1D signal."""
	signs = np.sign(signal)
	signs[signs == 0] = 1
	crossings = np.sum(signs[:-1] * signs[1:] < 0)
	return float(crossings / max(len(signal) - 1, 1))


def _load_json(path: Path) -> dict[str, Any]:
	"""Load JSON file with explicit error context."""
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except OSError as exc:
		raise RuntimeError(f"Unable to read JSON file {path}: {exc}") from exc
	except json.JSONDecodeError as exc:
		raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


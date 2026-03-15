"""RAFUI operator GUI for BLE connection, calibration, training, and live recognition."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, scrolledtext

from rafui_basic_print_vibecode import RafuiBLEClient, configure_logging
from rafui_recog_model_vibecode import (
    IDLE_LABEL,
    NOISE_LABEL,
    WINDOW_SIZE,
    detect_transition,
    load_model as load_model_v1,
    predict as predict_v1,
    save_model as save_model_v1,
    train as train_v1,
    extract_features,
)
from rafui_recog_model_2_vibecode import (
    load_model as load_model_v2,
    predict as predict_v2,
    save_model as save_model_v2,
    train_from_period_file,
)

matplotlib.use("TkAgg")

APP_TITLE = "RAFUI Operator"
JSON_DIR_NAME = "json"
MODEL_DIR_NAME = "models"
MODEL_GRAPH_DIR_NAME = "modelgraphs"
MODEL_FILE_PREFIX = "rafui_model"
MODEL2_FILE_PREFIX = "rafui_model2"
UI_POLL_MS = 80
PLOT_WINDOW_SECONDS = 5.0
MAX_LOG_LINES = 20
DEFAULT_IDLE_DURATION_S = 10.0
TRAINING_FILE_GLOB = "training_*.json"
PICKLE_GLOB = "*.pkl"
HARDWARE_VERSION = "1.0"
EXPECTED_SAMPLE_RATE_HZ = 50
CLUSTER_PLOT_MAX_POINTS = 250

STATE_DISPLAY_MAP = {
    IDLE_LABEL: "idle",
    "BTN_1": "but1",
    "BTN_2": "but2",
    "BTN_3": "but3",
    NOISE_LABEL: "idle",
}

STATE_COLORS = {
    IDLE_LABEL: "#f3f4f6",
    "BTN_1": "#fef3c7",
    "BTN_2": "#d1fae5",
    "BTN_3": "#dbeafe",
    NOISE_LABEL: "#fee2e2",
}


@dataclass(slots=True)
class UiEvent:
    """Thread-safe event payload from worker to GUI thread."""

    event_type: str
    payload: dict[str, Any]


class RafuiOperatorApp(tk.Tk):
    """Main RAFUI operator desktop application."""

    def __init__(self) -> None:
        super().__init__()
        configure_logging()

        self.title(APP_TITLE)
        self.geometry("1180x760")

        self.project_root = Path(__file__).resolve().parent
        self.json_dir = self.project_root / JSON_DIR_NAME
        self.model_dir = self.project_root / MODEL_DIR_NAME
        self.model_graph_dir = self.project_root / MODEL_GRAPH_DIR_NAME
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_graph_dir.mkdir(parents=True, exist_ok=True)

        self.ble_client: RafuiBLEClient | None = None
        self.idle_baseline: dict[str, float] | None = None
        self.model = None
        self.model_version = 1

        self.ui_queue: queue.Queue[UiEvent] = queue.Queue()
        self.stop_worker_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.sample_times_s: deque[float] = deque()
        self.sample_vmag: deque[float] = deque()
        self.sample_vph: deque[float] = deque()
        self.cluster_vmag: deque[float] = deque(maxlen=CLUSTER_PLOT_MAX_POINTS)
        self.cluster_vph: deque[float] = deque(maxlen=CLUSTER_PLOT_MAX_POINTS)
        self.cluster_states: deque[str] = deque(maxlen=CLUSTER_PLOT_MAX_POINTS)
        self.current_state = IDLE_LABEL
        self.current_confidence = 0.0
        self.is_busy = False
        self.training_capture_active = False
        self.training_capture_stop_event: threading.Event | None = None

        self._build_layout()
        self._set_phase_disconnected()
        self.after(UI_POLL_MS, self._drain_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        """Create all widgets and panels."""
        status_frame = tk.Frame(self, padx=10, pady=8)
        status_frame.pack(fill=tk.X)

        self.ble_status_var = tk.StringVar(value="BLE: Disconnected")
        self.state_var = tk.StringVar(value=f"State: {IDLE_LABEL}")
        self.conf_var = tk.StringVar(value="Confidence: 0.0%")
        self.touch_state_var = tk.StringVar(value="Touch: idle")

        tk.Label(status_frame, textvariable=self.ble_status_var, width=25, anchor="w").pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=self.state_var, width=20, anchor="w").pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(status_frame, textvariable=self.conf_var, width=20, anchor="w").pack(side=tk.LEFT, padx=(12, 0))

        button_frame = tk.Frame(self, padx=10, pady=4)
        button_frame.pack(fill=tk.X)

        self.connect_button = tk.Button(button_frame, text="Connect BLE", width=18, command=self._connect_ble)
        self.connect_button.pack(side=tk.LEFT, padx=4)

        self.calibrate_button = tk.Button(
            button_frame,
            text="Calibrate Idle",
            width=18,
            command=self._calibrate_idle,
        )
        self.calibrate_button.pack(side=tk.LEFT, padx=4)

        self.train_button = tk.Button(button_frame, text="Train Model", width=18, command=self._train_model)
        self.train_button.pack(side=tk.LEFT, padx=4)

        self.run_button = tk.Button(button_frame, text="Run Recognition", width=18, command=self._toggle_run)
        self.run_button.pack(side=tk.LEFT, padx=4)

        # Model 2 period capture buttons
        self.period_buttons = {}
        period_labels = ["IDLE", "BTN_1", "BTN_2", "BTN_3"]
        for label in period_labels:
            btn = tk.Button(button_frame, text=f"Record {label}", width=14, command=lambda l=label: self._record_period(l))
            btn.pack(side=tk.LEFT, padx=2)
            self.period_buttons[label] = btn

        self.finish_periods_button = tk.Button(button_frame, text="Finish Periods & Train", width=18, command=self._finish_periods_and_train)
        self.finish_periods_button.pack(side=tk.LEFT, padx=4)

        content = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left_panel = tk.Frame(content)
        right_panel = tk.Frame(content)
        content.add(left_panel, minsize=360)
        content.add(right_panel, minsize=760)

        tk.Label(left_panel, text="Event Log (latest 20)", anchor="w").pack(fill=tk.X)
        self.log_panel = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, height=38, state=tk.DISABLED)
        self.log_panel.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        right_container = tk.Frame(right_panel)
        right_container.pack(fill=tk.BOTH, expand=True)

        plot_frame = tk.Frame(right_container)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(right_container, padx=10, pady=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.figure = Figure(figsize=(8.4, 6.4), dpi=100)
        self.ax_raw = self.figure.add_subplot(211)
        self.ax_cluster = self.figure.add_subplot(212)

        self.ax_raw.set_title("Raw Signals (VMAG / VPH)")
        self.ax_raw.set_xlabel("Time (s)")
        self.ax_raw.set_ylabel("Voltage (V)")
        self.ax_raw.grid(alpha=0.25)

        self.vmag_line, = self.ax_raw.plot([], [], label="VMAG", color="#1d4ed8", linewidth=2.0)
        self.vph_line, = self.ax_raw.plot([], [], label="VPH", color="#b91c1c", linewidth=2.0)
        self.ax_raw.legend(loc="upper right")
        self.ax_raw.set_facecolor(STATE_COLORS[IDLE_LABEL])

        self.ax_cluster.set_title("Cluster Assignment (Current Point)")
        self.ax_cluster.set_xlabel("VMAG (V)")
        self.ax_cluster.set_ylabel("VPH (V)")
        self.ax_cluster.grid(alpha=0.25)
        self.ax_cluster.set_xlim(0.0, 3.3)
        self.ax_cluster.set_ylim(0.0, 3.3)

        self.cluster_scatter = self.ax_cluster.scatter([], [], s=20, alpha=0.45)
        self.cluster_current, = self.ax_cluster.plot(
            [],
            [],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=2,
            markeredgecolor="black",
            linestyle="None",
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tk.Label(info_frame, text="Current Touch", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        tk.Label(info_frame, textvariable=self.touch_state_var, font=("Segoe UI", 16, "bold")).pack(anchor="w")

    def _set_phase_disconnected(self) -> None:
        """Update button states for disconnected phase."""
        self.connect_button.config(state=tk.NORMAL)
        self.calibrate_button.config(state=tk.DISABLED)
        self.train_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)

    def _set_phase_connected(self) -> None:
        """Update button states once BLE is connected."""
        if self.is_busy:
            return
        self.connect_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.NORMAL)
        self.train_button.config(state=tk.DISABLED if self.idle_baseline is None else tk.NORMAL)
        self.run_button.config(state=tk.DISABLED if self.model is None else tk.NORMAL)

    def _set_phase_model_ready(self) -> None:
        """Update states after model is trained or loaded."""
        if self.is_busy:
            return
        self.connect_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.NORMAL)
        self.train_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)

    def _set_busy(self, busy: bool, *, reason: str | None = None) -> None:
        """Toggle busy UI mode while collecting/training in background."""
        self.is_busy = busy
        if busy:
            self.connect_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.DISABLED)
            if reason:
                self._append_log(reason)
            return

        if self.model is not None:
            self._set_phase_model_ready()
        elif self.ble_client is not None:
            self._set_phase_connected()
        else:
            self._set_phase_disconnected()

    def _set_phase_running(self, running: bool) -> None:
        """Disable setup actions while recognition is running."""
        if running:
            self.connect_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.DISABLED)
            self.run_button.config(text="Stop Recognition", state=tk.NORMAL)
        else:
            self.run_button.config(text="Run Recognition")
            if self.model is not None:
                self._set_phase_model_ready()
            elif self.ble_client is not None:
                self._set_phase_connected()
            else:
                self._set_phase_disconnected()

    def _connect_ble(self) -> None:
        """Attempt BLE connection in a short background worker."""

        def task() -> None:
            probe_client = RafuiBLEClient()
            try:
                import asyncio

                address = asyncio.run(probe_client.discover())
                # Keep only configuration data in GUI state; active BLE sessions are owned by worker loops.
                self.ble_client = RafuiBLEClient(address=address)
                self.ui_queue.put(UiEvent("ble_connected", {"address": address}))
            except Exception as exc:
                self.ui_queue.put(UiEvent("error", {"message": f"BLE connect failed: {exc}"}))

        threading.Thread(target=task, daemon=True).start()
        self._append_log("Scanning for RAFUI BLE peripheral...")

    def _calibrate_idle(self) -> None:
        """Capture idle baseline using BLE stream."""
        self._set_busy(True, reason="Idle calibration started (keep hand untouched).")
        self._clear_plot()

        def task() -> None:
            try:
                samples = asyncio.run(
                    self._capture_samples_async(
                        duration_s=DEFAULT_IDLE_DURATION_S,
                        mode_label="IDLE_CAPTURE",
                    )
                )
                baseline = self._compute_idle_baseline(samples)
                session_path = self._write_session_json(
                    session_type="idle",
                    samples=samples,
                    notes="Idle calibration capture",
                    extras={"baseline": baseline},
                )
                self.ui_queue.put(UiEvent("idle_done", {"baseline": baseline, "path": str(session_path)}))
            except Exception as exc:
                self.ui_queue.put(UiEvent("error", {"message": f"Idle calibration failed: {exc}"}))
            finally:
                self.ui_queue.put(UiEvent("busy_done", {}))

        threading.Thread(target=task, daemon=True).start()

    def _train_model(self) -> None:
        """Train model from selected training JSON file."""
        if self.training_capture_active:
            self._append_log("Stopping training capture...")
            if self.training_capture_stop_event is not None:
                self.training_capture_stop_event.set()
            self.train_button.config(text="Stopping...", state=tk.DISABLED)
            return

        selected_model_version = self._choose_model_version()
        if selected_model_version is None:
            return
        self.model_version = selected_model_version

        if self.model_version == 2:
            # Model 2: Use period capture buttons
            self._set_busy(False)
            self._clear_plot()
            self.period_records = []
            self.period_capture_active = True
            self._append_log("Model 2 period capture mode enabled. Use the buttons to record each period.")
            return

    def _record_period(self, label):
        if not hasattr(self, "period_capture_active") or not self.period_capture_active:
            self._append_log("Model 2 period capture not active.")
            return
        self._set_busy(True, reason=f"Capturing period: {label}...")
        self._clear_plot()
        stop_event = threading.Event()

        def task():
            try:
                samples = asyncio.run(
                    self._capture_samples_async(
                        mode_label=f"PERIOD_{label}",
                        stop_event=stop_event,
                    )
                )
            except Exception as exc:
                self._append_log(f"Period {label} capture failed: {exc}")
                self._set_busy(False)
                return
            elapsed = samples[-1]["t"] - samples[0]["t"] if samples else 0
            record = {
                "label": label,
                "started_at_iso": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ended_at_iso": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_s": elapsed / 1000.0,
                "sample_count": len(samples),
                "samples": samples,
            }
            if not hasattr(self, "period_records"):
                self.period_records = []
            self.period_records.append(record)
            self._set_busy(False)
            self._append_log(f"Period {label} recorded: {len(samples)} samples.")

        threading.Thread(target=task, daemon=True).start()

    def _finish_periods_and_train(self):
        if not hasattr(self, "period_capture_active") or not self.period_capture_active:
            self._append_log("Model 2 period capture not active.")
            return
        if not hasattr(self, "period_records") or not self.period_records:
            self._append_log("No periods recorded. Training aborted.")
            return

        # Stop any ongoing period capture
        if hasattr(self, "_period_capture_thread") and self._period_capture_thread is not None:
            try:
                self._period_capture_thread.join(timeout=1)
            except Exception:
                pass
            self._period_capture_thread = None

        self.period_capture_active = False
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        period_path = self.json_dir / f"periods_{stamp}.json"
        with open(period_path, "w", encoding="utf-8") as f:
            json.dump(self.period_records, f, indent=2)
        self._append_log(f"Periods saved to {period_path.name}")

        def task_model2() -> None:
            try:
                trained_model = train_from_period_file(str(period_path))
                model_path = self.model_dir / f"{MODEL2_FILE_PREFIX}_{stamp}.pkl"
                save_model_v2(trained_model, model_path)
                self.ui_queue.put(
                    UiEvent(
                        "train_done",
                        {
                            "model": trained_model,
                            "path": str(model_path),
                            "training_path": str(period_path),
                            "model_version": 2,
                        },
                    )
                )
            except Exception as exc:
                self.ui_queue.put(UiEvent("error", {"message": f"Training failed: {exc}"}))
            finally:
                self.ui_queue.put(UiEvent("busy_done", {}))

        self._set_busy(True, reason=f"Training Model 2 from {period_path.name}...")
        threading.Thread(target=task_model2, daemon=True).start()

        if self.idle_baseline is None:
            messagebox.showwarning("RAFUI", "Calibrate idle first.")
            return

        should_record = messagebox.askyesno(
            "RAFUI Training",
            "Record a new training session now?\n\n"
            "Yes: start capture now and stop it manually with the Train Model button\n"
            "when you finish touching BTN_1, BTN_2, BTN_3 in sequence.\n"
            "No: select an existing training JSON file.",
        )

        if should_record:
            self._set_training_capture_mode(True)
            self._append_log("Training capture started. Touch BTN_1 -> BTN_2 -> BTN_3 repeatedly, then press Train Model to stop.")
            self._clear_plot()
            self.training_capture_stop_event = threading.Event()

            def task_record_and_train() -> None:
                try:
                    samples = asyncio.run(
                        self._capture_samples_async(
                            mode_label="TRAINING_CAPTURE",
                            stop_event=self.training_capture_stop_event,
                        )
                    )
                    training_path = self._write_session_json(
                        session_type="training",
                        samples=samples,
                        notes="Captured in GUI: BTN_1 -> BTN_2 -> BTN_3 sequence",
                    )
                    self.ui_queue.put(UiEvent("training_recorded", {"path": str(training_path), "count": len(samples)}))

                    trained_model = train_v1(str(training_path), self.idle_baseline or {}, show_plot=False)
                    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    model_path = self.model_dir / f"{MODEL_FILE_PREFIX}_{stamp}.pkl"
                    save_model_v1(trained_model, model_path)
                    self.ui_queue.put(
                        UiEvent(
                            "train_done",
                            {
                                "model": trained_model,
                                "path": str(model_path),
                                "training_path": str(training_path),
                                "model_version": 1,
                            },
                        )
                    )
                except Exception as exc:
                    self.ui_queue.put(UiEvent("error", {"message": f"Training failed: {exc}"}))
                finally:
                    self.ui_queue.put(UiEvent("training_capture_finished", {}))
                    self.ui_queue.put(UiEvent("busy_done", {}))

            threading.Thread(target=task_record_and_train, daemon=True).start()
            return

        initial_dir = str(self.json_dir)
        path = filedialog.askopenfilename(
            title="Select training JSON",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        def task() -> None:
            try:
                trained_model = train_v1(path, self.idle_baseline or {}, show_plot=False)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                model_path = self.model_dir / f"{MODEL_FILE_PREFIX}_{stamp}.pkl"
                save_model_v1(trained_model, model_path)
                self.ui_queue.put(
                    UiEvent(
                        "train_done",
                        {
                            "model": trained_model,
                            "path": str(model_path),
                            "training_path": str(path),
                            "model_version": 1,
                        },
                    )
                )
            except Exception as exc:
                self.ui_queue.put(UiEvent("error", {"message": f"Training failed: {exc}"}))
            finally:
                self.ui_queue.put(UiEvent("busy_done", {}))

        self._set_busy(True, reason=f"Training from {Path(path).name}...")
        threading.Thread(target=task, daemon=True).start()

    def _toggle_run(self) -> None:
        """Start or stop live recognition."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_worker_event.set()
            if self.ble_client is not None:
                self.ble_client.stop_stream()
            self._append_log("Stopping recognition...")
            return

        if self.model is None:
            selected_model_version = self._choose_model_version()
            if selected_model_version is None:
                return
            self.model_version = selected_model_version

            latest_model = self._latest_model_path(self.model_version)
            if latest_model is None:
                messagebox.showwarning("RAFUI", "No model file found. Train model first.")
                return
            try:
                if self.model_version == 2:
                    self.model = load_model_v2(latest_model)
                else:
                    self.model = load_model_v1(latest_model)
                self._append_log(f"Loaded model v{self.model_version}: {latest_model.name}")
            except Exception as exc:
                messagebox.showerror("RAFUI", f"Failed to load model: {exc}")
                return

        self.stop_worker_event.clear()
        self.worker_thread = threading.Thread(target=self._run_recognition_worker, daemon=True)
        self.worker_thread.start()
        self._set_phase_running(True)
        self._append_log("Recognition started.")

    def _run_recognition_worker(self) -> None:
        """BLE + inference worker running outside Tk main thread."""
        assert self.model is not None

        address = self.ble_client.address if self.ble_client is not None else None
        local_client = RafuiBLEClient(address=address)
        prev_state = IDLE_LABEL
        rolling_samples: deque[dict[str, float]] = deque(maxlen=WINDOW_SIZE)

        def on_sample(sample: dict[str, float]) -> None:
            nonlocal prev_state
            if self.stop_worker_event.is_set():
                local_client.stop_stream()
                return
            rolling_samples.append(sample)
            inferred_state = IDLE_LABEL
            confidence = 0.0
            transition = None

            if len(rolling_samples) == WINDOW_SIZE:
                raw = np.array(
                    [[item["vmag"], item["vph"]] for item in rolling_samples],
                    dtype=np.float64,
                )
                if self.model_version == 2:
                    inferred_state, confidence = predict_v2(raw, self.model)
                else:
                    inferred_state, confidence = predict_v1(raw, self.model)
                transition = detect_transition(prev_state, inferred_state, confidence)
                prev_state = inferred_state

            self.ui_queue.put(
                UiEvent(
                    "sample",
                    {
                        "sample": sample,
                        "state": inferred_state,
                        "confidence": confidence,
                        "transition": transition,
                    },
                )
            )

        async def stream_task() -> None:
            try:
                await local_client.stream(on_sample)
            finally:
                await local_client.disconnect()

        try:
            import asyncio

            asyncio.run(stream_task())
        except Exception as exc:
            self.ui_queue.put(UiEvent("error", {"message": f"Recognition stream failed: {exc}"}))
        finally:
            self.ui_queue.put(UiEvent("worker_stopped", {}))

    def _drain_ui_queue(self) -> None:
        """Process worker events in main UI thread."""
        try:
            while True:
                event = self.ui_queue.get_nowait()
                self._handle_ui_event(event)
        except queue.Empty:
            pass

        self.after(UI_POLL_MS, self._drain_ui_queue)

    def _handle_ui_event(self, event: UiEvent) -> None:
        """Apply one UI event."""
        if event.event_type == "ble_connected":
            self.ble_status_var.set("BLE: Connected")
            address = str(event.payload.get("address", "unknown"))
            self._append_log(f"RAFUI detected at {address}.")
            self._set_phase_connected()
            return

        if event.event_type == "idle_done":
            self.idle_baseline = dict(event.payload["baseline"])
            save_path = str(event.payload.get("path", ""))
            self._append_log(f"Idle baseline saved to {Path(save_path).name}: {json.dumps(self.idle_baseline)}")
            self._set_phase_connected()
            self.train_button.config(state=tk.NORMAL)
            return

        if event.event_type == "training_recorded":
            self._append_log(
                f"Training data saved: {Path(str(event.payload['path'])).name} ({int(event.payload['count'])} samples)"
            )
            return

        if event.event_type == "train_done":
            self.model = event.payload["model"]
            self.model_version = int(event.payload.get("model_version", 1))
            self._append_log(f"Model trained and saved to {event.payload['path']}")
            self._set_phase_model_ready()
            training_path = str(event.payload.get("training_path", ""))
            if training_path and self.model_version == 1:
                self._show_training_cluster_plot(training_path)
            return

        if event.event_type == "sample":
            sample = event.payload["sample"]
            self._update_stream_plot(sample)
            state = str(event.payload["state"])
            confidence = float(event.payload["confidence"])
            self.current_state = state
            self.current_confidence = confidence
            self.state_var.set(f"State: {state}")
            self.conf_var.set(f"Confidence: {confidence * 100.0:.1f}%")
            self.touch_state_var.set(f"Touch: {STATE_DISPLAY_MAP.get(state, 'idle')}")
            self.ax_raw.set_facecolor(STATE_COLORS.get(state, STATE_COLORS[NOISE_LABEL]))
            self._update_cluster_plot(sample=sample, state=state)
            self.canvas.draw_idle()

            transition = event.payload.get("transition")
            if transition is not None:
                self._append_log(f"{transition}({state}) @ {sample['t']:.0f} ms")
            return

        if event.event_type == "worker_stopped":
            self._append_log("Recognition stopped.")
            self._set_phase_running(False)
            return

        if event.event_type == "busy_done":
            self._set_busy(False)
            return

        if event.event_type == "training_capture_finished":
            self._set_training_capture_mode(False)
            return

        if event.event_type == "error":
            message = str(event.payload.get("message", "Unknown error"))
            self._append_log(f"ERROR: {message}")
            messagebox.showerror("RAFUI", message)
            self._set_phase_running(False)

    def _update_stream_plot(self, sample: dict[str, float]) -> None:
        """Update rolling VMAG/VPH curves for last 5 seconds."""
        t_s = float(sample["t"]) / 1000.0
        self.sample_times_s.append(t_s)
        self.sample_vmag.append(float(sample["vmag"]))
        self.sample_vph.append(float(sample["vph"]))

        cutoff = t_s - PLOT_WINDOW_SECONDS
        while self.sample_times_s and self.sample_times_s[0] < cutoff:
            self.sample_times_s.popleft()
            self.sample_vmag.popleft()
            self.sample_vph.popleft()

        xs = np.array(self.sample_times_s, dtype=np.float64)
        if xs.size == 0:
            return
        xs = xs - xs[0]

        self.vmag_line.set_data(xs, np.array(self.sample_vmag, dtype=np.float64))
        self.vph_line.set_data(xs, np.array(self.sample_vph, dtype=np.float64))

        self.ax_raw.set_xlim(max(0.0, float(xs.min())), max(PLOT_WINDOW_SECONDS, float(xs.max()) + 0.2))

        y_all = np.concatenate(
            [
                np.array(self.sample_vmag, dtype=np.float64),
                np.array(self.sample_vph, dtype=np.float64),
            ]
        )
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        margin = max((y_max - y_min) * 0.2, 0.05)
        self.ax_raw.set_ylim(y_min - margin, y_max + margin)

    def _update_cluster_plot(self, *, sample: dict[str, float], state: str) -> None:
        """Update VMAG-VPH scatter and highlight currently assigned state point."""
        vmag = float(sample["vmag"])
        vph = float(sample["vph"])
        self.cluster_vmag.append(vmag)
        self.cluster_vph.append(vph)
        self.cluster_states.append(state)

        self.ax_cluster.cla()
        self.ax_cluster.set_title("Cluster Assignment (Current Point)")
        self.ax_cluster.set_xlabel("VMAG (V)")
        self.ax_cluster.set_ylabel("VPH (V)")
        self.ax_cluster.grid(alpha=0.25)

        if self.cluster_vmag:
            x = np.array(self.cluster_vmag, dtype=np.float64)
            y = np.array(self.cluster_vph, dtype=np.float64)
            colors = [STATE_COLORS.get(label, STATE_COLORS[NOISE_LABEL]) for label in self.cluster_states]
            self.ax_cluster.scatter(x, y, c=colors, s=20, alpha=0.45)
            self.ax_cluster.scatter([x[-1]], [y[-1]], s=100, facecolors="none", edgecolors="black", linewidths=2)

            x_min, x_max = float(np.min(x)), float(np.max(x))
            y_min, y_max = float(np.min(y)), float(np.max(y))
            x_margin = max((x_max - x_min) * 0.2, 0.05)
            y_margin = max((y_max - y_min) * 0.2, 0.05)
            self.ax_cluster.set_xlim(max(0.0, x_min - x_margin), min(3.3, x_max + x_margin))
            self.ax_cluster.set_ylim(max(0.0, y_min - y_margin), min(3.3, y_max + y_margin))
        else:
            self.ax_cluster.set_xlim(0.0, 3.3)
            self.ax_cluster.set_ylim(0.0, 3.3)

    def _append_log(self, message: str) -> None:
        """Append one message and keep only latest N lines."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_panel.configure(state=tk.NORMAL)
        self.log_panel.insert(tk.END, line)
        all_lines = self.log_panel.get("1.0", tk.END).splitlines()
        if len(all_lines) > MAX_LOG_LINES:
            trimmed = "\n".join(all_lines[-MAX_LOG_LINES:]) + "\n"
            self.log_panel.delete("1.0", tk.END)
            self.log_panel.insert(tk.END, trimmed)
        self.log_panel.configure(state=tk.DISABLED)
        self.log_panel.see(tk.END)

    def _latest_model_path(self, model_version: int) -> Path | None:
        """Return most recent model pickle path if available."""
        prefix = MODEL2_FILE_PREFIX if model_version == 2 else MODEL_FILE_PREFIX
        files = sorted(self.model_dir.glob(f"{prefix}_*.pkl"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _choose_model_version(self) -> int | None:
        """Ask user to choose model 1 or model 2.

        Returns:
            1 for model 1, 2 for model 2, None for cancel.
        """
        answer = messagebox.askyesnocancel(
            "Select Training/Run Model",
            "Choose modeling approach:\n\n"
            "Yes = Model 1 (current window clustering)\n"
            "No = Model 2 (period-based clustering)\n"
            "Cancel = abort",
        )
        if answer is None:
            return None
        return 1 if answer else 2

    def _clear_plot(self) -> None:
        """Reset rolling plot buffers and redraw empty axes."""
        self.sample_times_s.clear()
        self.sample_vmag.clear()
        self.sample_vph.clear()
        self.vmag_line.set_data([], [])
        self.vph_line.set_data([], [])
        self.cluster_vmag.clear()
        self.cluster_vph.clear()
        self.cluster_states.clear()
        self.ax_raw.set_xlim(0.0, PLOT_WINDOW_SECONDS)
        self.ax_raw.set_ylim(0.0, 3.3)
        self.ax_raw.set_facecolor(STATE_COLORS[IDLE_LABEL])
        self.ax_cluster.cla()
        self.ax_cluster.set_title("Cluster Assignment (Current Point)")
        self.ax_cluster.set_xlabel("VMAG (V)")
        self.ax_cluster.set_ylabel("VPH (V)")
        self.ax_cluster.grid(alpha=0.25)
        self.ax_cluster.set_xlim(0.0, 3.3)
        self.ax_cluster.set_ylim(0.0, 3.3)
        self.touch_state_var.set("Touch: idle")
        self.canvas.draw_idle()

    async def _capture_samples_async(
        self,
        *,
        mode_label: str,
        duration_s: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> list[dict[str, float]]:
        """Capture BLE stream samples until duration elapses or stop_event is set."""
        if duration_s is None and stop_event is None:
            raise ValueError("Either duration_s or stop_event must be provided")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be > 0")

        address = self.ble_client.address if self.ble_client is not None else None
        local_client = RafuiBLEClient(address=address)
        samples: list[dict[str, float]] = []
        start_time = time.monotonic()

        def on_sample(sample: dict[str, float]) -> None:
            samples.append(sample)
            self.ui_queue.put(
                UiEvent(
                    "sample",
                    {
                        "sample": sample,
                        "state": IDLE_LABEL,
                        "confidence": 0.0,
                        "transition": None,
                    },
                )
            )
            elapsed = time.monotonic() - start_time
            if duration_s is not None and elapsed >= duration_s:
                local_client.stop_stream()
            if stop_event is not None and stop_event.is_set():
                local_client.stop_stream()

        monitor_task: asyncio.Task[None] | None = None

        async def monitor_stop_request() -> None:
            while stop_event is not None and not stop_event.is_set():
                await asyncio.sleep(0.05)
            if stop_event is not None and stop_event.is_set():
                local_client.stop_stream()

        try:
            if stop_event is not None:
                monitor_task = asyncio.create_task(monitor_stop_request())
            await local_client.stream(on_sample)
        finally:
            if monitor_task is not None:
                monitor_task.cancel()
            await local_client.disconnect()

        if not samples:
            raise RuntimeError(f"{mode_label} collected 0 samples")
        return samples

    def _set_training_capture_mode(self, active: bool) -> None:
        """Switch UI to/from manual training capture mode."""
        self.training_capture_active = active
        if active:
            self.is_busy = True
            self.connect_button.config(state=tk.DISABLED)
            self.calibrate_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.DISABLED)
            self.train_button.config(text="Stop Training Capture", state=tk.NORMAL)
            return

        self.training_capture_stop_event = None
        self.train_button.config(text="Train Model")

    def _compute_idle_baseline(self, samples: list[dict[str, float]]) -> dict[str, float]:
        """Compute idle baseline statistics from sample list."""
        values = np.array([[s["vmag"], s["vph"]] for s in samples], dtype=np.float64)
        return {
            "mean_vmag": float(np.mean(values[:, 0])),
            "std_vmag": float(np.std(values[:, 0])),
            "mean_vph": float(np.mean(values[:, 1])),
            "std_vph": float(np.std(values[:, 1])),
        }

    def _write_session_json(
        self,
        *,
        session_type: str,
        samples: list[dict[str, float]],
        notes: str,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Write session payload under json folder with standard schema fields."""
        timestamp_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        timestamp_file = datetime.now(UTC).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")

        payload: dict[str, Any] = {
            "session": session_type,
            "timestamp": timestamp_iso,
            "hardware_version": HARDWARE_VERSION,
            "sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
            "notes": notes,
            "samples": samples,
        }
        if extras:
            payload.update(extras)

        output_path = self.json_dir / f"{session_type}_{timestamp_file}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def _show_training_cluster_plot(self, training_path: str) -> None:
        """Show training cluster scatter after model training completes."""
        if self.model is None:
            return

        try:
            payload = json.loads(Path(training_path).read_text(encoding="utf-8"))
            samples = payload.get("samples", [])
            if len(samples) < WINDOW_SIZE:
                self._append_log("Skipping cluster plot: not enough training samples.")
                return

            values = np.array([[float(s["vmag"]), float(s["vph"])] for s in samples], dtype=np.float64)
            rows: list[np.ndarray] = []
            for start in range(0, len(values) - WINDOW_SIZE + 1):
                rows.append(extract_features(values[start : start + WINDOW_SIZE]))
            if not rows:
                self._append_log("Skipping cluster plot: no feature windows generated.")
                return

            feature_matrix = np.vstack(rows)
            feature_scaled = self.model.scaler.transform(feature_matrix)

            deltas = feature_scaled[:, None, :] - self.model.medoid_centers_scaled[None, :, :]
            cluster_idx = np.argmin(np.linalg.norm(deltas, axis=2), axis=1)

            plt.figure("RAFUI Training Clusters", figsize=(8, 6))
            plt.clf()
            for idx in range(self.model.medoid_centers_unscaled.shape[0]):
                mask = cluster_idx == idx
                label = self.model.label_map.get(int(idx), f"CLUSTER_{idx}")
                plt.scatter(feature_matrix[mask, 0], feature_matrix[mask, 2], s=12, alpha=0.6, label=label)

            medoids = self.model.medoid_centers_unscaled
            plt.scatter(medoids[:, 0], medoids[:, 2], s=260, marker="*", color="black", label="Medoids")
            plt.title("RAFUI Training Clusters (mean_vmag vs mean_vph)")
            plt.xlabel("mean_vmag")
            plt.ylabel("mean_vph")
            plt.legend()
            plt.tight_layout()
            timestamp_file = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            graph_path = self.model_graph_dir / f"cluster_graph_{timestamp_file}.png"
            plt.savefig(graph_path, dpi=180)
            plt.show(block=False)
            self._append_log(f"Displayed training cluster plot and saved to {graph_path}.")
        except Exception as exc:
            self._append_log(f"Cluster plot display failed: {exc}")

    def _on_close(self) -> None:
        """Graceful shutdown of background worker and app."""
        self.stop_worker_event.set()
        if self.ble_client is not None:
            self.ble_client.stop_stream()
        self.destroy()


def main() -> None:
    """Launch RAFUI operator application."""
    app = RafuiOperatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()

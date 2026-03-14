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
    load_model,
    predict,
    save_model,
    train,
)

matplotlib.use("TkAgg")

APP_TITLE = "RAFUI Operator"
JSON_DIR_NAME = "json"
MODEL_DIR_NAME = "models"
MODEL_FILE_PREFIX = "rafui_model"
UI_POLL_MS = 80
PLOT_WINDOW_SECONDS = 5.0
MAX_LOG_LINES = 20
DEFAULT_IDLE_DURATION_S = 10.0
DEFAULT_TRAINING_DURATION_S = 20.0
TRAINING_FILE_GLOB = "training_*.json"
PICKLE_GLOB = "*.pkl"
HARDWARE_VERSION = "1.0"
EXPECTED_SAMPLE_RATE_HZ = 50

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
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.ble_client: RafuiBLEClient | None = None
        self.idle_baseline: dict[str, float] | None = None
        self.model = None

        self.ui_queue: queue.Queue[UiEvent] = queue.Queue()
        self.stop_worker_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.sample_times_s: deque[float] = deque()
        self.sample_vmag: deque[float] = deque()
        self.sample_vph: deque[float] = deque()
        self.current_state = IDLE_LABEL
        self.current_confidence = 0.0
        self.is_busy = False

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

        content = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left_panel = tk.Frame(content)
        right_panel = tk.Frame(content)
        content.add(left_panel, minsize=360)
        content.add(right_panel, minsize=760)

        tk.Label(left_panel, text="Event Log (latest 20)", anchor="w").pack(fill=tk.X)
        self.log_panel = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, height=38, state=tk.DISABLED)
        self.log_panel.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.figure = Figure(figsize=(8.2, 4.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("VMAG / VPH Rolling Window")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid(alpha=0.25)

        self.vmag_line, = self.ax.plot([], [], label="VMAG", color="#1d4ed8", linewidth=2.0)
        self.vph_line, = self.ax.plot([], [], label="VPH", color="#b91c1c", linewidth=2.0)
        self.ax.legend(loc="upper right")
        self.ax.set_facecolor(STATE_COLORS[IDLE_LABEL])

        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        if self.idle_baseline is None:
            messagebox.showwarning("RAFUI", "Calibrate idle first.")
            return

        should_record = messagebox.askyesno(
            "RAFUI Training",
            "Record a new training session now?\n\n"
            "Yes: capture ~20s while touching BTN_1, BTN_2, BTN_3 in sequence.\n"
            "No: select an existing training JSON file.",
        )

        if should_record:
            self._set_busy(True, reason="Training capture started. Touch BTN_1 -> BTN_2 -> BTN_3 repeatedly.")
            self._clear_plot()

            def task_record_and_train() -> None:
                try:
                    samples = asyncio.run(
                        self._capture_samples_async(
                            duration_s=DEFAULT_TRAINING_DURATION_S,
                            mode_label="TRAINING_CAPTURE",
                        )
                    )
                    training_path = self._write_session_json(
                        session_type="training",
                        samples=samples,
                        notes="Captured in GUI: BTN_1 -> BTN_2 -> BTN_3 sequence",
                    )
                    self.ui_queue.put(UiEvent("training_recorded", {"path": str(training_path), "count": len(samples)}))

                    trained_model = train(str(training_path), self.idle_baseline or {})
                    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    model_path = self.model_dir / f"{MODEL_FILE_PREFIX}_{stamp}.pkl"
                    save_model(trained_model, model_path)
                    self.ui_queue.put(UiEvent("train_done", {"model": trained_model, "path": str(model_path)}))
                except Exception as exc:
                    self.ui_queue.put(UiEvent("error", {"message": f"Training failed: {exc}"}))
                finally:
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
                trained_model = train(path, self.idle_baseline or {})
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                model_path = self.model_dir / f"{MODEL_FILE_PREFIX}_{stamp}.pkl"
                save_model(trained_model, model_path)
                self.ui_queue.put(UiEvent("train_done", {"model": trained_model, "path": str(model_path)}))
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
            latest_model = self._latest_model_path()
            if latest_model is None:
                messagebox.showwarning("RAFUI", "No model file found. Train model first.")
                return
            try:
                self.model = load_model(latest_model)
                self._append_log(f"Loaded model: {latest_model.name}")
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

        local_client = RafuiBLEClient()
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
                inferred_state, confidence = predict(raw, self.model)
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
            self._append_log(f"Model trained and saved to {event.payload['path']}")
            self._set_phase_model_ready()
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
            self.ax.set_facecolor(STATE_COLORS.get(state, STATE_COLORS[NOISE_LABEL]))
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

        self.ax.set_xlim(max(0.0, float(xs.min())), max(PLOT_WINDOW_SECONDS, float(xs.max()) + 0.2))

        y_all = np.concatenate(
            [
                np.array(self.sample_vmag, dtype=np.float64),
                np.array(self.sample_vph, dtype=np.float64),
            ]
        )
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        margin = max((y_max - y_min) * 0.2, 0.05)
        self.ax.set_ylim(y_min - margin, y_max + margin)

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

    def _latest_model_path(self) -> Path | None:
        """Return most recent model pickle path if available."""
        files = sorted(self.model_dir.glob(PICKLE_GLOB), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def _clear_plot(self) -> None:
        """Reset rolling plot buffers and redraw empty axes."""
        self.sample_times_s.clear()
        self.sample_vmag.clear()
        self.sample_vph.clear()
        self.vmag_line.set_data([], [])
        self.vph_line.set_data([], [])
        self.ax.set_xlim(0.0, PLOT_WINDOW_SECONDS)
        self.ax.set_ylim(0.0, 3.3)
        self.ax.set_facecolor(STATE_COLORS[IDLE_LABEL])
        self.canvas.draw_idle()

    async def _capture_samples_async(self, duration_s: float, mode_label: str) -> list[dict[str, float]]:
        """Capture BLE stream samples for finite duration and mirror live data to UI."""
        if duration_s <= 0:
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
            if time.monotonic() - start_time >= duration_s:
                local_client.stop_stream()

        try:
            await local_client.stream(on_sample)
        finally:
            await local_client.disconnect()

        if not samples:
            raise RuntimeError(f"{mode_label} collected 0 samples")
        return samples

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

"""BLE client utilities for RAFUI sample streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bleak import BleakClient, BleakScanner

LOGGER = logging.getLogger(__name__)

DEVICE_NAME = "RAFUI"
AUTH_MESSAGE = "AUTH:RAFUI123"

# Replace these UUIDs with your firmware values if needed.
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
TX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
RX_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"

AUTH_TIMEOUT_S = 3.0
POST_AUTH_SETTLE_S = 0.15
MAX_RECONNECT_RETRIES = 5
RECONNECT_BASE_DELAY_S = 0.5
MAX_RECONNECT_DELAY_S = 8.0
COMMAND_WRITE_WITH_RESPONSE = False

VMAG_MIN = 0.0
VMAG_MAX = 3.3
VPH_MIN = 0.0
VPH_MAX = 3.3


@dataclass(slots=True)
class StreamSample:
	"""Typed representation of one RAFUI sample."""

	t: float
	vmag: float
	vph: float

	def as_dict(self) -> dict[str, float]:
		"""Convert sample into the JSON-compatible dictionary form."""
		return {"t": self.t, "vmag": self.vmag, "vph": self.vph}


class RafuiBLEClient:
	"""Async BLE transport client for RAFUI peripheral communication."""

	def __init__(self, address: str | None = None) -> None:
		self._address = address
		self._client: BleakClient | None = None
		self._stop_event = asyncio.Event()
		self._line_buffer = ""
		self._connected_event = asyncio.Event()

	@property
	def is_connected(self) -> bool:
		"""Return True when BLE client is connected."""
		return bool(self._client and self._client.is_connected)

	@property
	def address(self) -> str | None:
		"""Return configured BLE device address, when known."""
		return self._address

	async def discover(self, *, device_name: str = DEVICE_NAME) -> str:
		"""Discover RAFUI peripheral and return its BLE address."""
		address = await self._discover_device_address(device_name)
		self._address = address
		return address

	async def connect(self) -> None:
		"""Connect to RAFUI BLE peripheral and perform shared-secret auth."""
		if self.is_connected:
			LOGGER.debug("BLE already connected")
			return

		target_address = self._address
		if target_address is None:
			target_address = await self._discover_device_address(DEVICE_NAME)
			self._address = target_address

		try:
			self._client = BleakClient(target_address, disconnected_callback=self._on_disconnected)
		except TypeError:
			# Older bleak versions may not expose disconnected_callback in constructor.
			self._client = BleakClient(target_address)
		await self._client.connect()
		LOGGER.info("Connected to RAFUI peripheral at %s", target_address)

		try:
			await asyncio.wait_for(self.send_command(AUTH_MESSAGE), timeout=AUTH_TIMEOUT_S)
		except Exception as exc:
			LOGGER.exception("Authentication write failed: %s", exc)
			await self.disconnect()
			raise

		self._connected_event.set()
		LOGGER.info("Shared-secret auth sent")
		await asyncio.sleep(POST_AUTH_SETTLE_S)

	async def disconnect(self) -> None:
		"""Disconnect from RAFUI peripheral safely."""
		self._stop_event.set()
		if not self._client:
			return

		try:
			if self._client.is_connected:
				try:
					await self.send_command("STOP")
				except Exception:
					LOGGER.debug("STOP command failed during disconnect", exc_info=True)
				await self._client.disconnect()
				LOGGER.info("Disconnected from RAFUI peripheral")
		finally:
			self._connected_event.clear()
			self._client = None

	async def send_command(self, command: str) -> None:
		"""Send command to peripheral over RX characteristic."""
		if not self._client or not self._client.is_connected:
			raise RuntimeError("BLE client is not connected")
		payload = command.encode("utf-8")
		try:
			await self._client.write_gatt_char(
				RX_CHAR_UUID,
				payload,
				response=COMMAND_WRITE_WITH_RESPONSE,
			)
		except Exception:
			# Retry once with opposite write mode for backend compatibility.
			await self._client.write_gatt_char(
				RX_CHAR_UUID,
				payload,
				response=not COMMAND_WRITE_WITH_RESPONSE,
			)
		LOGGER.debug("Sent command: %s", command)

	def stop_stream(self) -> None:
		"""Request graceful stop of stream loop."""
		self._stop_event.set()

	async def stream(
		self,
		callback: Callable[[dict[str, float]], None],
		*,
		auto_start: bool = True,
	) -> None:
		"""Stream JSON samples from peripheral and invoke callback per sample.

		Args:
			callback: Function invoked for each validated sample.
			auto_start: If True, sends START command after notifications are enabled.
		"""
		retries = 0
		self._stop_event.clear()

		while not self._stop_event.is_set() and retries <= MAX_RECONNECT_RETRIES:
			try:
				await self.connect()
				assert self._client is not None

				await self._client.start_notify(TX_CHAR_UUID, self._make_notification_handler(callback))
				LOGGER.info("Notifications started")

				if auto_start:
					await self.send_command("START")

				await self._wait_until_disconnected_or_stopped()

				if self._stop_event.is_set():
					LOGGER.info("Stream stop requested")

				await self._client.stop_notify(TX_CHAR_UUID)
				if self._stop_event.is_set():
					break

				retries += 1
				await self._backoff_sleep(retries)
			except Exception as exc:
				retries += 1
				LOGGER.exception("Stream error (attempt %d/%d): %s", retries, MAX_RECONNECT_RETRIES, exc)
				if retries > MAX_RECONNECT_RETRIES:
					break
				await self._backoff_sleep(retries)
			finally:
				if self._client and not self._client.is_connected:
					self._client = None

		if retries > MAX_RECONNECT_RETRIES and not self._stop_event.is_set():
			raise RuntimeError("Exceeded BLE reconnect retries")

	async def _wait_until_disconnected_or_stopped(self) -> None:
		"""Block stream loop until stop is requested or BLE disconnects."""
		while not self._stop_event.is_set():
			if not self.is_connected:
				LOGGER.warning("BLE disconnected")
				self._connected_event.clear()
				return
			await asyncio.sleep(0.1)

	async def _discover_device_address(self, device_name: str) -> str:
		"""Find peripheral address by advertised name."""
		LOGGER.info("Scanning for BLE device named %s", device_name)

		# Prefer richer advertisement data when available.
		try:
			discovered = await BleakScanner.discover(timeout=5.0, return_adv=True)
		except TypeError:
			discovered = None

		if discovered:
			for device, adv_data in discovered.values():
				name = (device.name or adv_data.local_name or "").strip()
				service_uuids = [str(item).lower() for item in (adv_data.service_uuids or [])]
				if name == device_name or SERVICE_UUID.lower() in service_uuids:
					LOGGER.info(
						"Found RAFUI candidate at %s (name=%s, service_match=%s)",
						device.address,
						name or "<unnamed>",
						SERVICE_UUID.lower() in service_uuids,
					)
					return str(device.address)

			visible = []
			for device, adv_data in discovered.values():
				name = device.name or adv_data.local_name or "<unnamed>"
				uuids = ",".join(adv_data.service_uuids or [])
				visible.append(f"{name} ({device.address}) [uuids={uuids or '-'}]")
			visible_text = ", ".join(visible) if visible else "no devices discovered"
			raise RuntimeError(f"Device '{device_name}' not found; visible: {visible_text}")

		devices = await BleakScanner.discover(timeout=5.0)
		for device in devices:
			if (device.name or "").strip() == device_name:
				LOGGER.info("Found %s at %s", device_name, device.address)
				return str(device.address)

		visible = [f"{d.name or '<unnamed>'} ({d.address})" for d in devices]
		visible_text = ", ".join(visible) if visible else "no devices discovered"
		raise RuntimeError(f"Device '{device_name}' not found; visible: {visible_text}")

	def _on_disconnected(self, _: Any) -> None:
		"""Default disconnect callback."""
		LOGGER.warning("Disconnected from peripheral")
		self._connected_event.clear()

	async def _backoff_sleep(self, retry_count: int) -> None:
		"""Sleep with exponential backoff between reconnect attempts."""
		delay = min(RECONNECT_BASE_DELAY_S * (2 ** (retry_count - 1)), MAX_RECONNECT_DELAY_S)
		LOGGER.info("Reconnecting in %.2fs", delay)
		await asyncio.sleep(delay)

	def _make_notification_handler(
		self,
		callback: Callable[[dict[str, float]], None],
	) -> Callable[[Any, bytearray], None]:
		"""Build notification callback that parses newline-delimited JSON."""

		def _handler(_: Any, data: bytearray) -> None:
			try:
				chunk = data.decode("utf-8", errors="ignore")
				self._line_buffer += chunk
				while "\n" in self._line_buffer:
					line, self._line_buffer = self._line_buffer.split("\n", 1)
					line = line.strip()
					if not line:
						continue
					parsed = self._validate_and_parse_sample(line)
					if parsed is None:
						continue
					callback(parsed.as_dict())
			except Exception:
				LOGGER.exception("Failed handling notification payload")

		return _handler

	def _validate_and_parse_sample(self, line: str) -> StreamSample | None:
		"""Parse and validate one JSON sample line."""
		try:
			payload = json.loads(line)
		except json.JSONDecodeError:
			LOGGER.warning("Dropped invalid JSON: %s", line)
			return None

		required_keys = {"t", "vmag", "vph"}
		if not required_keys.issubset(payload.keys()):
			LOGGER.warning("Dropped malformed sample keys: %s", payload)
			return None

		try:
			t_value = float(payload["t"])
			vmag_value = float(payload["vmag"])
			vph_value = float(payload["vph"])
		except (TypeError, ValueError):
			LOGGER.warning("Dropped sample with non-numeric fields: %s", payload)
			return None

		if t_value < 0:
			LOGGER.warning("Dropped sample with negative timestamp: %s", payload)
			return None
		if not (VMAG_MIN <= vmag_value <= VMAG_MAX):
			LOGGER.warning("Dropped sample with out-of-range VMAG: %s", payload)
			return None
		if not (VPH_MIN <= vph_value <= VPH_MAX):
			LOGGER.warning("Dropped sample with out-of-range VPH: %s", payload)
			return None

		return StreamSample(t=t_value, vmag=vmag_value, vph=vph_value)


def configure_logging(level: int = logging.INFO) -> None:
	"""Configure default logging for RAFUI BLE tools."""
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


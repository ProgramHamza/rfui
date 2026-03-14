"""RAFUI ESP32-S3 MicroPython BLE peripheral implementation."""

import asyncio
import ujson
import utime
from machine import ADC, Pin

import aioble
import bluetooth

DEBUG = True

DEVICE_NAME = "RAFUI"
AUTH_MESSAGE = "AUTH:RAFUI123"
AUTH_TIMEOUT_S = 3.0

SERVICE_UUID = bluetooth.UUID("12345678-1234-5678-1234-56789abcdef0")
TX_CHAR_UUID = bluetooth.UUID("12345678-1234-5678-1234-56789abcdef1")
RX_CHAR_UUID = bluetooth.UUID("12345678-1234-5678-1234-56789abcdef2")

GPIO_VMAG = 4
GPIO_VPH = 5
ADC_ATTENUATION = ADC.ATTN_11DB
ADC_AVERAGE_COUNT = 4
ADC_MAX = 4095
ADC_REF_VOLTAGE = 3.3

SAMPLE_RATE_HZ = 50
SAMPLE_PERIOD_S = 1.0 / SAMPLE_RATE_HZ


def log(message):
	"""Print debug logs when DEBUG is enabled."""
	if DEBUG:
		print("[RAFUI]", message)


def adc_to_voltage(raw_value):
	"""Convert 12-bit ADC raw reading to voltage."""
	return raw_value * (ADC_REF_VOLTAGE / ADC_MAX)


def read_adc_average(adc):
	"""Read ADC multiple times and return averaged raw value."""
	total = 0
	for _ in range(ADC_AVERAGE_COUNT):
		total += adc.read()
	return total // ADC_AVERAGE_COUNT


def parse_written_payload(result):
	"""Normalize aioble written() return value into text payload.

	aioble may return either raw bytes, or a tuple like (connection, data).
	"""
	data = result
	if isinstance(result, tuple):
		# Most aioble builds return (connection, data)
		data = result[1] if len(result) > 1 else b""
	if data is None:
		return ""
	if isinstance(data, str):
		return data.strip()
	try:
		return bytes(data).decode().strip()
	except Exception:
		return ""


uart_service = aioble.Service(SERVICE_UUID)
tx_characteristic = aioble.Characteristic(uart_service, TX_CHAR_UUID, notify=True)
rx_characteristic = aioble.Characteristic(
	uart_service,
	RX_CHAR_UUID,
	write=True,
	capture=True,
)
aioble.register_services(uart_service)

vmag_adc = ADC(Pin(GPIO_VMAG))
vph_adc = ADC(Pin(GPIO_VPH))
vmag_adc.atten(ADC_ATTENUATION)
vph_adc.atten(ADC_ATTENUATION)


async def require_auth(connection):
	"""Require shared-secret auth command within timeout or disconnect."""
	try:
		result = await asyncio.wait_for(rx_characteristic.written(), AUTH_TIMEOUT_S)
	except asyncio.TimeoutError:
		log("Auth timeout")
		await connection.disconnect()
		return False

	message = parse_written_payload(result)
	log("Auth payload: {}".format(message or "<empty>"))

	if message != AUTH_MESSAGE:
		log("Auth failed")
		await connection.disconnect()
		return False

	log("Auth successful")
	return True


async def command_loop(connection, state):
	"""Process incoming commands over RX characteristic."""
	while connection.is_connected():
		try:
			result = await rx_characteristic.written()
			command = parse_written_payload(result).upper()
			if not command:
				continue
			if command == "START":
				state["streaming"] = True
				log("Streaming enabled")
			elif command == "STOP":
				state["streaming"] = False
				log("Streaming disabled")
			elif command == "PING":
				tx_characteristic.notify(connection, b"PONG\n")
				log("PONG sent")
			elif command == AUTH_MESSAGE:
				# Ignore repeated auth command after session is already authenticated.
				pass
			else:
				log("Unknown command: {}".format(command))
		except Exception as exc:
			log("Command loop error: {}".format(exc))
			await asyncio.sleep(0.05)


async def stream_loop(connection, state):
	"""Send RAFUI samples at ~50Hz while streaming is active."""
	while connection.is_connected():
		if state["streaming"]:
			t_ms = utime.ticks_ms()
			vmag_raw = read_adc_average(vmag_adc)
			vph_raw = read_adc_average(vph_adc)
			sample = {
				"t": t_ms,
				"vmag": round(adc_to_voltage(vmag_raw), 6),
				"vph": round(adc_to_voltage(vph_raw), 6),
			}
			payload = ujson.dumps(sample) + "\n"
			try:
				tx_characteristic.notify(connection, payload.encode())
			except Exception as exc:
				log("Notify error: {}".format(exc))

		await asyncio.sleep(SAMPLE_PERIOD_S)


async def peripheral_loop():
	"""Main BLE peripheral lifecycle loop."""
	while True:
		log("Advertising as {}".format(DEVICE_NAME))
		connection = await aioble.advertise(
			250000,
			name=DEVICE_NAME,
			services=[SERVICE_UUID],
			appearance=0,
		)

		log("Client connected")
		session_state = {"streaming": False}
		authed = await require_auth(connection)
		if not authed:
			session_state["streaming"] = False
			continue

		stream_task = asyncio.create_task(stream_loop(connection, session_state))
		command_task = asyncio.create_task(command_loop(connection, session_state))

		try:
			while connection.is_connected():
				await asyncio.sleep(0.1)
		finally:
			session_state["streaming"] = False
			stream_task.cancel()
			command_task.cancel()
			log("Client disconnected; returning to advertise")


async def main():
	"""Entrypoint for RAFUI peripheral firmware."""
	await peripheral_loop()


try:
	asyncio.run(main())
finally:
	asyncio.new_event_loop()


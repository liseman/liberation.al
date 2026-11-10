# Historical hardware candidates

These are candidates from the October 2026 wearable design, not a verified
shopping list or a statement of what is installed. Exact part numbers, electrical
ratings, connectors, quantities, and availability need checking before purchase or
assembly. The [original plan](../history/plan-2026-10.md) explains their intended
roles; [Architecture](../ARCHITECTURE.md) describes the running software.

| Area | Candidates considered | Design question |
|---|---|---|
| Main compute | Apple Silicon desktop carried in a pack | Power supply, cooling, and mounting |
| Satellite compute | Raspberry Pi 5 | Whether depth mapping needs a separate computer |
| Sensor controller | Teensy 4.1 | Sensor buses, firmware, and synchronization |
| Voice input | Throat microphone and wireless USB receiver | Intelligibility and end-to-end latency |
| Ambient audio | ReSpeaker XVF3800 USB microphone array | Echo cancellation, direction of arrival, and mounting |
| Voice output | Bone-conduction transducer, with an external amplifier if needed | Impedance, power, output level, and feedback |
| Vision | Three overlapping Arducam IMX291 fisheye cameras | Calibration, USB bandwidth, and mechanical placement |
| Depth | OAK-D camera | Model-specific capabilities and synchronization |
| Projection | Portable projector and optional motorized mount | Throw, power, brightness, and surface calibration |
| Body sensing | IMU, temperature/humidity sensor, and light sensor | Electrical compatibility and useful sample rates |
| Haptics | Small vibration motors and a suitable driver | Power budget and safe drive circuitry |
| Structure | Frame, enclosures, cables, and powered USB hub | Weight, strain relief, cooling, and weather protection |

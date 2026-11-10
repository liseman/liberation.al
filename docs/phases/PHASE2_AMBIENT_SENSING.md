# Phase 2: Ambient Sensing Integration

**Historical proposal, October 2026. Not implemented.** The hardware claims,
API examples, and performance targets below were not validated. The code sketches
are not installed tools. See [Architecture](../ARCHITECTURE.md) for the current
system and [Roadmap](../ROADMAP.md) for planned work.

## Overview

Integrate Teensy 4.1 microcontroller as sensor hub for ambient data collection. Real-time IMU, temperature/humidity, light, and haptic feedback to enrich silent I/O context and enable environmental awareness.

## Hardware

### Teensy 4.1 Core
- **MCU:** ARM Cortex-M7, 600 MHz, 1024 KB SRAM
- **Connection:** USB-C to the Mac (data + power)
- **Bus:** I2C (3.3V pullups) for multi-sensor comms

### Sensor Suite

**9-axis IMU (MPU-9250 or ICM-20948)**
- ±16g accelerometer, ±2000°/s gyroscope, ±4800 µT magnetometer
- Detects motion, orientation, gestures
- Use case: Activate/deactivate silent I/O on body motion

**Temperature & Humidity (DHT22 or BME680)**
- ±0.5°C accuracy, 2-5% RH
- I2C address: 0x77 (BME680)
- Use case: Environmental context for responses

**Ambient Light (TSL2561)**
- 16-bit ADC, full-spectrum + IR channels
- Adaptive brightness for projected AR
- I2C address: 0x29 or 0x39

**Haptic Feedback (3mm coin vibration motors)**
- 2× motors, 3.3V 4mA each
- Driven via GPIO PWM
- Use case: Silent notifications, confirmation feedback

## Firmware Architecture

**Teensy Sketch (TeensyDuino IDE or PlatformIO)**

```cpp
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_TSL2561_U.h>
#include <Adafruit_BME680.h>

// I2C Addresses
#define IMU_ADDR 0x68
#define BME_ADDR 0x77
#define TSL_ADDR 0x29

struct SensorPacket {
  uint32_t timestamp_ms;
  float accel_x, accel_y, accel_z;
  float gyro_x, gyro_y, gyro_z;
  float temp_c, humidity_pct, pressure_pa;
  uint16_t lux;
};

// Global sensor objects
Adafruit_MPU6050 mpu;
Adafruit_BME680 bme;
Adafruit_TSL2561_U tsl(TSL2561_ADDR_FLOAT);

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Initialize sensors
  if (!mpu.begin()) Serial.println("MPU6050 init failed");
  if (!bme.begin(BME_ADDR)) Serial.println("BME680 init failed");
  if (!tsl.begin()) Serial.println("TSL2561 init failed");
  
  // Configure IMU for 100 Hz sample rate
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_100_HZ);
  
  Serial.println("Sensors ready");
}

void loop() {
  static unsigned long last_sample = 0;
  unsigned long now = millis();
  
  // Sample at 100 Hz (~10 ms per sample)
  if (now - last_sample < 10) return;
  last_sample = now;
  
  SensorPacket packet;
  packet.timestamp_ms = now;
  
  // Read IMU
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);
  packet.accel_x = accel.acceleration.x;
  packet.accel_y = accel.acceleration.y;
  packet.accel_z = accel.acceleration.z;
  packet.gyro_x = gyro.gyro.x;
  packet.gyro_y = gyro.gyro.y;
  packet.gyro_z = gyro.gyro.z;
  
  // Read BME680
  if (bme.performReading()) {
    packet.temp_c = bme.temperature;
    packet.humidity_pct = bme.humidity;
    packet.pressure_pa = bme.pressure;
  }
  
  // Read TSL2561
  sensors_event_t event;
  tsl.getEvent(&event);
  packet.lux = event.light;
  
  // Serialize and send via USB
  Serial.write((uint8_t*)&packet, sizeof(packet));
}
```

## Mac Integration (Python)

**phase2_sensor_daemon.py**

```python
import serial
import struct
import json
from collections import deque
from threading import Thread

class TeensySensorHub:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.packet_fmt = "I6f3fH"  # timestamp_ms, 6 floats (accel+gyro), 3 floats (temp/humid/press), uint16 (lux)
        self.packet_size = struct.calcsize(self.packet_fmt)
        
        self.sensor_buffer = deque(maxlen=100)  # Rolling 1s buffer @ 100 Hz
        self.running = True
        
    def read_loop(self):
        """Continuous sensor read from Teensy"""
        while self.running:
            if self.ser.in_waiting >= self.packet_size:
                raw = self.ser.read(self.packet_size)
                try:
                    data = struct.unpack(self.packet_fmt, raw)
                    packet = {
                        "ts": data[0],
                        "ax": data[1], "ay": data[2], "az": data[3],
                        "gx": data[4], "gy": data[5], "gz": data[6],
                        "temp_c": data[7], "humidity": data[8], "pressure": data[9],
                        "lux": data[10],
                    }
                    self.sensor_buffer.append(packet)
                except struct.error:
                    pass
    
    def get_latest(self):
        """Latest sensor packet"""
        return self.sensor_buffer[-1] if self.sensor_buffer else None
    
    def get_motion_state(self):
        """Compute motion magnitude from IMU"""
        if len(self.sensor_buffer) < 10:
            return {"motion": 0}
        
        recent = list(self.sensor_buffer)[-10:]
        ax = [p["ax"] for p in recent]
        ay = [p["ay"] for p in recent]
        az = [p["az"] for p in recent]
        
        motion = sum((ax[i]**2 + ay[i]**2 + az[i]**2)**0.5 for i in range(len(ax))) / len(ax)
        return {"motion": motion, "is_stationary": motion < 0.5}
    
    def haptic_pulse(self, motor: int, duration_ms: int = 100, intensity: float = 1.0):
        """Trigger haptic feedback via Teensy GPIO"""
        # Command: 'H' + motor_id (0-1) + duration_ms (2 bytes) + intensity (0-255)
        cmd = bytes([ord('H'), motor, duration_ms & 0xFF, (duration_ms >> 8) & 0xFF, int(intensity * 255)])
        self.ser.write(cmd)

if __name__ == "__main__":
    hub = TeensySensorHub()
    Thread(target=hub.read_loop, daemon=True).start()
    
    try:
        while True:
            latest = hub.get_latest()
            motion = hub.get_motion_state()
            if latest:
                print(f"Temp: {latest['temp_c']:.1f}°C | Light: {latest['lux']} lux | Motion: {motion['motion']:.2f}g")
            time.sleep(0.5)
    except KeyboardInterrupt:
        hub.running = False
```

## Integration Points

- **Silent I/O activation:** Motion-based wake (gyro > threshold) triggers ASR pipeline
- **Environmental context:** Ambient light → projected UI brightness; temperature → response tone
- **Haptic feedback:** Confirmation pulses on TTS completion; warning vibrations on obstacle detection
- **SLAM preparation:** IMU data pre-processed for odometry estimation (Phase 5)

## Success Criteria

- Teensy firmware stable at 100 Hz sampling
- USB serial comms reliable, < 5ms read latency
- IMU calibration converged (zero-g offset < ±0.1g)
- Haptic feedback perceptible at 50-100 ms pulse width
- Environmental data enriches LLM context without exceeding sub-700ms latency

## Next: Phase 3

Projection mapping (Pico 4 + OAK-D) with surface calibration and AR overlay.

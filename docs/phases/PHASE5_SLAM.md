# Phase 5: SLAM & Odometry

**Historical proposal, October 2026. Not implemented.** The hardware claims,
API examples, and performance targets below were not validated. The code sketches
are not installed tools. See [Architecture](../ARCHITECTURE.md) for the current
system and [Roadmap](../ROADMAP.md) for planned work.

## Overview

Deploy ORB-SLAM3 on Raspberry Pi 5 satellite for real-time visual inertial odometry. Fuse OAK-D stereo depth + IMU for accurate 6-DOF pose estimation and environmental mapping. Enable spatial memory and context-aware navigation.

## Hardware

### Compute
- **Raspberry Pi 5:** 8GB RAM, arm64 (secondary compute)
- **Connected to:** the Mac via USB for data sync
- **Role:** Dedicated SLAM processor (offload from the Mac)

### Sensors
- **OAK-D stereo camera:** 640×400 @ 30 FPS depth, USB to Pi 5
- **Teensy IMU:** 100 Hz 9-axis data, via I2C bridge to Pi
- **Fusion:** Depth + gyro for visual-inertial SLAM

## ORB-SLAM3 Stack

### Installation on Raspberry Pi 5

**Dependencies:**
```bash
# Ubuntu 22.04 for Pi 5
sudo apt update
sudo apt install -y cmake git libopencv-dev libeigen3-dev libpangolin-dev libboost-all-dev

# ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh

# OAK-D SDK
pip3 install depthai opencv-python numpy scipy
```

### Initialization: Stereo Calibration

ORB-SLAM3 requires intrinsic & extrinsic calibration. Use OAK-D's built-in factory calibration:

```python
import depthai as dai

# Query OAK-D calibration
pipeline = dai.Pipeline()
device = dai.Device()
calibData = device.readCalibration()

# Extract matrices
K_left = calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
K_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT)
R = calibData.getStereoRightRectificationRotation()
T = calibData.getBaseline() * 0.001  # Baseline in meters

# Save to ORB-SLAM3 config
yaml_config = f"""
Camera.fx: {K_left[0][0]}
Camera.fy: {K_left[1][1]}
Camera.cx: {K_left[0][2]}
Camera.cy: {K_left[1][2]}
Camera.k1: {calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)[0]}
# ... full YAML
"""
```

### C++ SLAM Node (Pi 5)

```cpp
#include <iostream>
#include <System.h>
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // Initialize ORB-SLAM3
    ORB_SLAM3::System SLAM("ORBvoc.txt", "StereoConfig.yaml", ORB_SLAM3::System::STEREO);
    
    // OAK-D pipeline
    dai::Pipeline pipeline;
    auto stereo = pipeline.createStereoDepth();
    auto spatialDetectionNetwork = pipeline.createYoloSpatialDetectionPostProcessor();
    
    // Connect device
    dai::Device device(pipeline);
    
    // Main loop
    while (true) {
        auto inRight = device.getOutputQueue("right")->get<dai::ImgFrame>();
        auto inLeft = device.getOutputQueue("left")->get<dai::ImgFrame>();
        auto inDepth = device.getOutputQueue("depth")->get<dai::ImgFrame>();
        
        cv::Mat leftMat = inLeft->getCvFrame();
        cv::Mat rightMat = inRight->getCvFrame();
        
        // Track frame + depth
        cv::Mat Tcw = SLAM.TrackStereo(leftMat, rightMat, inLeft->getTimestamp().count());
        
        // Publish pose to Mac via USB
        PublishPose(Tcw);
    }
    
    SLAM.Shutdown();
    return 0;
}
```

## Pose & Map Fusion (Mac)

**phase5_slam_fusion.py** - Receive pose updates + map vertices:

```python
import socket
import json
import numpy as np
from collections import deque

class SLAMFusionHub:
    def __init__(self):
        self.pose_buffer = deque(maxlen=30)  # Last 1s @ 30 Hz
        self.map_points = []
        self.keyframes = []
        self.listening = True
    
    def listen_from_pi(self, host="raspberrypi.local", port=5555):
        """Receive SLAM poses over network."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        
        while self.listening:
            data, addr = sock.recvfrom(1024)
            pose_msg = json.loads(data.decode())
            
            # Parse 4×4 pose matrix
            Tcw = np.array(pose_msg["Tcw"])
            
            self.pose_buffer.append({
                "timestamp": pose_msg["ts"],
                "Tcw": Tcw,
                "translation": Tcw[:3, 3],
                "rotation": Tcw[:3, :3],
            })
    
    def get_current_pose(self):
        """Latest pose estimate."""
        return self.pose_buffer[-1] if self.pose_buffer else None
    
    def get_trajectory(self):
        """Smoothed trajectory from pose buffer."""
        positions = np.array([p["translation"] for p in self.pose_buffer])
        # Optionally apply Kalman filtering or smoothing
        return positions
    
    def estimate_depth(self, screen_2d_point):
        """Project 2D screen point to 3D world via SLAM map."""
        # Use map points near camera + pose estimate
        pass
```

## Integration: Spatial Memory

**Anchor AR objects in world coordinates:**

```python
def place_ar_annotation(text: str, world_position: np.ndarray):
    """
    Persistent 3D annotation in SLAM coordinate frame.
    
    Example: Place "coffee station" marker at discovered location.
    """
    slam = SLAMFusionHub()
    current_pose = slam.get_current_pose()
    
    if not current_pose:
        print("❌ SLAM not initialized")
        return
    
    # Transform annotation to world frame
    Tcw = current_pose["Tcw"]
    Twc = np.linalg.inv(Tcw)
    
    # Annotation in camera frame (small offset forward)
    camera_pos = np.array([0, 0, 1])  # 1m forward
    world_pos = Twc @ np.append(camera_pos, 1)
    
    # Store
    db = AnnotationDB()
    db.add_annotation({
        "text": text,
        "world_position": world_pos[:3],
        "timestamp": time.time(),
        "source_pose_id": current_pose["timestamp"],
    })
    
    print(f"✓ Placed '{text}' at {world_pos[:3]}")
```

## Success Criteria

- ORB-SLAM3 initialization <5 seconds (stereo + IMU calibration)
- Pose estimation <50ms per frame (30 FPS stereo input)
- Map density ≥500 point features per frame in textured environments
- Trajectory drift <5% over 100m path (audited via ground truth)
- Spatial annotations persist across sessions (database lookup by proximity)
- Integration with Phase 3 projection: project spatial UI anchors aligned to SLAM map

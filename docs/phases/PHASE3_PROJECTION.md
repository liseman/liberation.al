# Phase 3: Projection Mapping & Depth

**Historical proposal, October 2026. Not implemented.** The hardware claims,
API examples, and performance targets below were not validated. The code sketches
are not installed tools. See [Architecture](../ARCHITECTURE.md) for the current
system and [Roadmap](../ROADMAP.md) for planned work.

## Overview

Mount Pico 4 projector on motorized gimbal + OAK-D stereo depth camera for real-time AR projection mapping. Capabilities:

- AR overlay on surfaces (walls, tables, floors)
- Obstacle detection
- Spatial UI anchoring
- Surface geometry warping (dynamic homography)

## Hardware

### Projector
- **Pico 4:** 1200 ANSI lumens, 2160×1200 DLP, 0.25-2.5m throw
- **Gimbal:** 2-axis motorized (pan/tilt, ±45° range)
- **Control:** USB from the Mac or direct servo commands

### Depth Camera
- **OAK-D:** Stereo + ToF, 30-80 FPS, 0.1-10m range, 640×400 depth resolution
- **Connection:** USB to the Mac
- **Latency:** <35ms depth frame

## Calibration

**Camera-projector extrinsics:**
- Checkerboard pattern (10×7 grid, known square size)
- Capture depth + projector image simultaneously
- Compute intrinsic matrices (K_depth, K_projector)
- Solve for extrinsic rotation/translation (R, t)
- Store calibration for depth-to-projector homography

**Runtime warping:**
```python
# Compute homography from depth map
H = compute_homography_from_depth(depth_frame, calibration)

# Warp AR content to surface
warped_image = cv2.warpPerspective(ar_content, H, (width, height))

# Project onto surface
projector.display(warped_image)
```

## Integration Points

- **Depth → Obstacle detection:** Flag regions where depth < threshold
- **Depth → AR placement:** Anchor 3D objects to detected surfaces
- **Gimbal → Dynamic framing:** Pan/tilt to follow face or gesture
- **LLM → Spatial annotations:** Render text/UI anchored to objects in scene

## Success Criteria

- Camera-projector calibration error <5 pixels
- Depth capture + warp pipeline <100ms
- Projections align with surfaces within ±1cm at 2m distance
- Real-time gimbal pan/tilt responsive to scene changes

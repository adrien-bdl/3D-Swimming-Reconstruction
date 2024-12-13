# 3D Swimming Reconstruction

A system for creating 3D animations of swimming movements using video capture, computer vision, and Blender.

## Prerequisites

- Blender (3D animation software)
- Two USB-compatible cameras for stereo vision
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

### Blender Resources (`/Blender`)
- **`data/`**: Swimming motion capture data
  - `network_solution_2_Amel_Melih_202312191406_Depart_compet_Crawl.tsv`: Elite athlete data (X,Y coordinates)
  - `our_swim_data.dat`: Custom swim data (X,Y,Z coordinates) generated using 2.5D OpenCV method
- **`meshes/`**: 3D models and rigging assets
- **`scripts/`**
  - `run_anim_Amel_Melih.py`: Animation generator for elite athlete data
  - `run_anim_our_swim.py`: Animation generator for custom swim data
  - `utils.py`: Helper functions and utilities

### 2.5D Reconstruction (`/2.5D`)
- **`INSEP/`**: INSEP's X,Y coordinate extraction tools
- **`laser_3D/`**: Per-camera capture data
- **`OpenCV_reconstruction/`**
  - `bodypose3d/`: Joint position detection system
  - `GoPro/`: Camera capture data
  - `stereo_calibration/`: Camera calibration tools

## Usage Guide

### 1. Camera Calibration

Connect both cameras to your computer via USB and run:
```bash
python 2.5D/OpenCV_reconstruction/stereo_calibration/calib.py \
    2.5D/OpenCV_reconstruction/stereo_calibration/calibration_settings.yaml
```

**Note**: Ensure cameras are firmly mounted and don't move during calibration.

### 2. 3D Coordinate Generation

Generate 3D coordinates from video footage:
```bash
python 2.5D/OpenCV_reconstruction/bodypose3d/bodypose3d.py
```

Output file location: `2.5D/OpenCV_reconstruction/bodypose3d/out/kpts_3d.dat`

### 3. Preview 3D Coordinates

View the generated 3D motion data:
```bash
python 2.5D/OpenCV_reconstruction/bodypose3d/show_3d_pose.py
```

### 4. Blender Animation

1. Copy the generated coordinates:
   ```bash
   cp 2.5D/OpenCV_reconstruction/bodypose3d/out/kpts_3d.dat Blender/data/
   ```

2. Update the path in `Blender/scripts/run_anim_our_swim.py`:
   ```python
   KEYPOINTS_PATH = "../data/kpts_3d.dat"
   ```

3. Launch Blender and execute the animation script.


## Acknowledgments

- INSEP and Prof. Rémi Carmignani for providing their support, base code, and initial data
- [python_stereo_camera_calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate) for the calibration
- [Real time 3D body pose estimation using MediaPipe](https://github.com/TemugeB/bodypose3d) for coordinates generation and visualization
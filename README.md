# Cave Explorer 🛰️

Autonomous exploration of a Mars-like cave with a ROS 2 / Gazebo rover.
The robot maps the cave with frontier-based exploration, detects and localises artefacts with a Faster R-CNN detector that stays usable under simulated dust and darkness, inspects interesting finds at close range, and analyses the cave geometry (wide chambers, narrow passages, bottlenecks) in real time.

> Team project for **49274 Space Robotics** (UTS) by Julien Labie, Noah Meissner and Dorian Veloso.

▶️ **[Watch the demo video](https://drive.google.com/file/d/1DZUfOD2u6j2-tVyr4uhz-rBHjEbr6-rS/view?usp=sharing)**


---

## Highlights

| | Result (single full run from the start pose) |
|---|---|
| 🗺️ Exploration | **1,701 m²** mapped, no frontier left, in **529 s** sim time / 350 m, then returned home |
| 🔍 Inspection | **9 / 9** artefacts (3 aliens, 3 balls, 3 mushrooms) inspected on the first attempt, heading error ≤ 8.3° |
| 🧠 Detection | **mAP50 0.949** (test set), no false alarms on negative images |
| 🧱 Recovery | 2 collisions with lidar-invisible crystals, both recovered automatically |
| 📐 Geometry | Widest area 11.4 m, bottleneck 2.47 m (only connection), analysis in **~13 ms** |

*Results come from one validated run and a small test set (17 images); they show that the system works, not a statistical evaluation.*

---

## Features

### Perception
- **Dataset collection** – `DatasetSaver` node stores synchronised RGB + 16-bit depth images, with time and duplicate filtering.
- **Artefact detection** – Faster R-CNN (MobileNetV3-Large-FPN, COCO-pretrained) fine-tuned on 6 classes: alien, ball, graffiti, mushroom, stone, stop sign. Smaller anchors (16–256 px) for distant objects; runs on CPU.
- **3D localisation** – box centre + depth (25 % quantile) → pinhole projection → two-stage TF (`odom→camera` at image time, `map→odom` latest) → line-of-sight check against the SLAM map → DBSCAN clustering (eps 1.5 m, min 3 observations).

### Planning
- **Frontier exploration** – frontiers ranked by `U = size − travel distance − |heading change|`, with travel distance from a wavefront over the occupancy grid. Goals are re-evaluated every 2 s while driving.
- **Stuck recovery** – distinguishes a collision (rewind along the trajectory + virtual obstacle on `/virtual_obstacles`) from a Nav2 planning failure (goal dropped).
- **Close-range inspection** – 24 candidate viewpoints on a 2 m circle, filtered by wall clearance, reachability and line of sight; final heading alignment with the camera.
- **Behaviour switching** – state machine Explore → Inspect → (retry once / abandon) → Resume → Return home, with visibility-aware loss detection and a CSV log of every inspection.

### Advanced
- **Robust perception** – realistic lens dust and low-light/noise degradations applied online, ramping up over the mission; detector retrained with degraded augmentation.
- **Cave geometry analysis** – distance transform + gradient/Hessian ridge detection: widest chambers, narrow passages, saddle points, and a cut test to find true chokepoints.

---

## Architecture

```
            /camera/image, /camera/depth/image, /camera/camera_info
                                   │
                                   ▼
┌───────────────────────── cave_explorer.py ─────────────────────────┐
│ degrade image ─► detector ─► localisation ─► DBSCAN artefacts      │
│                                                   │                │
│   state machine:  EXPLORE_FRONTIERS ◄──► INSPECT_ARTEFACT          │
│   (frontier_exploration.py)              (inspection.py)           │
└──────────────┬──────────────────────────────────────┬──────────────┘
               │ Nav2 NavigateToPose                   │ markers / images
               ▼                                       ▼
          Nav2 + slam_toolbox                         RViz
               ▲
     /virtual_obstacles (costmap source)

cave_geometry.py  ◄── /map + TF ──►  /distance_transform, /cave_geometry_markers
```

### Main topics

| Topic | Type | Description |
|---|---|---|
| `degraded_image` | `sensor_msgs/Image` | Camera image after dust/darkness degradation |
| `detections_image` | `sensor_msgs/Image` | Image annotated with detections |
| `/virtual_obstacles` | `sensor_msgs/PointCloud2` <!-- TODO: check type --> | Obstacles added after collisions with lidar-invisible rocks |
| `goal_pose_vis` | `geometry_msgs/PoseStamped` | Current goal (visualisation only, avoids clashing with Nav2's `goal_pose`) |
| `/distance_transform` | `nav_msgs/OccupancyGrid` | Distance-to-wall layer |
| `/cave_geometry_markers` | `visualization_msgs/MarkerArray` | Ridge, wide areas, passages, bottleneck |

---

## Installation

**Requirements:** Ubuntu 22.04, ROS 2 Humble, Gazebo (Ignition) with `ros_ign_gazebo` / `ros_gz_bridge`, Nav2, slam_toolbox.

```bash
# clone into your workspace
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone <repo-url>

# ROS dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Python dependencies (CPU is enough)
pip install torch torchvision opencv-python scikit-learn torchmetrics

colcon build --symlink-install
source install/setup.bash
```

**Model weights:** place the trained detector in `TODO/path/model.pth` <!-- TODO: path or download link -->.

---

## Usage

Start each launch file in its own terminal (source the workspace in each) and wait until the previous one is up:

```bash
# 1. Gazebo world + robot
ros2 launch cave_explorer cave_explorer_startup.launch.py

# 2. SLAM + Nav2
ros2 launch cave_explorer cave_explorer_navigation.launch.py

# 3. Autonomy (exploration, detection, inspection, geometry)
ros2 launch cave_explorer cave_explorer_autonomy.launch.py
```

### Launch parameters

| Parameter | Default | Description |
|---|---|---|
| `geometry` | `True` | Start the cave geometry analysis node |
| degradation mode <!-- TODO: exact name --> | `none` | `dirt`, `dark` or `both` |
| `degrade_ramp_time` | `300` | Seconds of sim time until degradation reaches full strength |

Example: `ros2 launch cave_explorer cave_explorer_autonomy.launch.py geometry:=True <mode>:=both`

### Outputs
- Close-range image of every inspected artefact + `inspections.csv` (result, reason, attempts, duration, distance, heading error)
- `~/cave_geometry.png` – distance transform with ridge, widest area and passages

---

## RViz legend

| Marker | Meaning |
|---|---|
| Coloured spheres + label | Localised artefacts (alien green, ball blue, graffiti magenta, mushroom orange, stone brown, stop red) |
| Disc under artefact | Inspection status: orange pending · yellow in progress · green inspected · red abandoned |
| Coloured clusters / numbers | Frontier clusters and their utility; green sphere = chosen frontier |
| Circle + green/red points | Inspection standoff circle with valid/rejected viewpoints |
| Yellow points | Medial axis (ridge) of the cave |
| Green disc | Widest accessible area |
| Red line | Bottleneck; orange lines = other passages |

---

## Training the detector

```bash
# TODO: replace with your actual scripts
python3 scripts/train_detector.py --data <coco_export> --epochs 20 [--degrade]
python3 scripts/evaluate.py --model <model.pth> --degrade dust60+dark0.66
```

Dataset: 111 annotated images (+15 negatives) from the simulation, annotated in Roboflow, exported as COCO, split 70/15/15. Training: SGD, lr 0.005, cosine schedule, batch size 2, ~25 min on a laptop CPU.

---

## Project structure

```
cave_explorer/
├── cave_explorer/
│   ├── cave_explorer.py          # main node: perception, state machine
│   ├── frontier_exploration.py   # frontier detection & utility (NumPy/OpenCV, ROS-free)
│   ├── inspection.py             # viewpoint selection
│   ├── cave_geometry.py          # geometry analysis node
│   └── ...                       # TODO: dataset saver, degradation, training
├── launch/
├── config/                       # Nav2 params (incl. virtual obstacle source)
└── README.md
```

---

## Known limitations

- Low crystals are invisible to the 1 m-high lidar; the robot recovers from collisions instead of avoiding them.
- Small test set (17 images), and train/test images come from the same runs, so detection scores are optimistic.
- Artefacts are tracked by class and proximity, without persistent IDs.
- An artefact first detected while returning home is not inspected.
- The geometry node uses a 0.4 m robot radius, Nav2 uses 0.5 m.

## Future work

Camera-coverage frontiers, information-gain term, persistent artefact tracking (Hungarian matching + Kalman filter), behaviour tree instead of hand-written state flags, and a topology graph of the cave from a proper skeleton.

---

## Team

| Member | Contribution |
|---|---|
| Noah Meissner | Perception 1–3, Robust perception |
| Dorian Veloso | Frontier exploration, close-range inspection |
| Julien Labie | Behaviour switching, cave geometry analysis, videos |


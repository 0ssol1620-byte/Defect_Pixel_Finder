# Defect Pixel Finder

A PyQt5-based GUI tool that detects **dark/bright defect pixels** in camera images and exports the results as CSV.

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/5b605817-72b9-4e89-b5f3-57a5008c557f" />

---

## Main Features

- Detects dark and bright defect pixels from a single frame
- Supports 8-bit and 10/12/14/16-bit images (Bayer RAW, mono, and RGB formats)
- Integrates with Euresys Coaxlink + eGrabber cameras  
  (or runs in file-only mode when no camera is available)
- Visualizes defect locations as overlays on the image
- Exports defect pixel lists to CSV

---

## Project Structure

```text
Defect_Pixel_Finder/
├─ main.py                  # Application entry point
├─ defect_tool_ui.py        # PyQt5 GUI (main window)
├─ defect_detection.py      # Defect pixel detection algorithms
├─ draw_overlay.py          # Overlay drawing utilities for defect visualization
├─ core/
│  ├─ camera_controller.py  # eGrabber-based camera control
│  ├─ camera_exceptions.py  # Camera-related exception definitions
│  ├─ camera_facade.py      # Qt QObject wrapper (CxpCamera)
│  └─ controller_pool.py    # CameraController pool/registry
└─ workers/
   └─ grab_worker.py        # Background frame-drain worker thread
```

---

## Module Overview

### `main.py`

- Parses command line arguments:
  - `--log-level {DEBUG,INFO,WARNING,ERROR}`
  - `--highdpi` – force Qt High-DPI scaling
  - `--icon` – path to the window/taskbar icon file
- Creates the `QApplication`, instantiates the `DefectTool` widget, and starts the event loop

---

### `defect_tool_ui.py`

Implements the main GUI for defect pixel inspection.

- **Buttons / Controls**
  - **Connect** – search for and connect to the first available camera
  - **Snap** – capture a single frame
  - **Open Image** – load an image from file
  - **Dark Search / Bright Search** – run defect detection on the current image
  - **Export CSV** – save detected defect pixels to a CSV file

- **Parameters**
  - **Block size** – tile size used for local statistics
  - **Dark level (DN)** – threshold offset for dark pixels
  - **Bright (±%)** – percentage deviation threshold relative to the local average
  - **PixelFormat (Load)** – assumed pixel format when loading images from file
  - **Histogram normalize / stretch** – display-time histogram normalization/stretch

Frames from the camera or images loaded from disk are passed to the algorithms in
`defect_detection.py` to get dark/bright defect results.  
`draw_overlay.py` is used to draw rectangles/markers on top of the image for visualization.

---

### `defect_detection.py`

Contains the main defect pixel detection logic.

Key responsibilities:

- Convert Bayer / mono / RGB formats to a common grayscale representation
- Compute local averages using block statistics and interpolation
- Decide whether each pixel is dark or bright relative to its local context
- Generate clusters (regions) from a binary defect mask and return rectangles,
  counts, and lists of points

Representative functions include:

- **Format / Preprocessing**
  - `bayer_to_flat(...)`, `bayer_flatfield_equalize(...)`
  - `rgb_flatfield_equalize(...)`
  - Histogram normalization utilities
- **Defect Detection**
  - `FindDarkFieldClusterRect_...`
  - `FindBrightFieldClusterRect_...`
- **Mask → Cluster**
  - Helper functions that convert masks into clustered regions

---

### `draw_overlay.py`

- `DefectLoc` – data class representing a single overlay item (color, line width, rectangle, label)
- `DrawFigure` – manages multiple `DefectLoc` items and renders them onto the original image

The GUI uses this module to display detection results on top of the image.

---

### `core` Package

#### `camera_controller.py`

- Controls cameras via the Euresys eGrabber SDK
- Discovers cameras, connects/disconnects, starts/stops grabbing, and receives frames
- Emits signals when new frames arrive so that upper layers can react
- Provides dummy classes when the eGrabber module is unavailable so that the
  application can still run without a physical camera

#### `camera_facade.py`

- Defines the `CxpCamera` Qt `QObject` wrapper
- Internally uses `CameraController` and exposes a simpler interface for the UI layer

#### `controller_pool.py`

- Manages a global pool of `CameraController` instances
- Registers controllers with IDs and supports lookup and broadcast operations

#### `camera_exceptions.py`

- Defines custom exception classes for camera-related errors

---

### `workers/grab_worker.py`

- A `QThread`-based background worker
- Continuously drains frames from the camera and forwards them upstream
- Helps prevent driver buffer overruns by keeping the queue empty

---

## Installation & Execution

### 1) Dependencies

- Python 3.8 or later (recommended)
- Required packages:
  - `PyQt5`
  - `numpy`
  - `opencv-python`
- Optional (for camera integration):
  - Euresys Coaxlink frame grabber
  - Euresys eGrabber SDK and Python bindings

Example:

```bash
pip install PyQt5 numpy opencv-python
# Install eGrabber and its Python module using the official Euresys installer
```

### 2) Run

```bash
python main.py --log-level INFO --highdpi
```

or:

```bash
python -m main
```

Options:

- `--log-level` : one of `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `--highdpi`   : force High-DPI scaling on high-resolution monitors
- `--icon`      : icon file to use for the window/taskbar

---

## Usage Examples

### With a Camera

1. Launch the application  
2. Click **Connect** to detect and connect to a camera  
3. Click **Snap** to capture one frame  
4. Adjust Block size, Dark level, and Bright%  
5. Click **Dark Search** / **Bright Search** to detect defect pixels  
6. Inspect the overlay; export results to CSV if needed  

### Without a Camera (Image File Only)

1. Launch the application  
2. Click **Open Image** to select a test image  
3. Set **PixelFormat (Load)** to match the actual image format  
4. Adjust the parameters and run Dark/Bright Search  
5. Save the results as CSV  

---

## CSV Format

The CSV output is roughly as follows (actual headers may differ depending on implementation):

```text
x,y   # each line represents one defect pixel coordinate (column, row)
10,15
100,200
...
```

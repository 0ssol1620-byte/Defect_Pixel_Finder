# Defect Pixel Finder

**Defect Pixel Finder** is a PyQt5-based desktop tool that helps you **find dark and bright defect pixels**
in camera images – and export the results as a simple CSV file.

You can:

- Connect to a CoaXPress camera (via Euresys eGrabber) **or** just load image files
- Adjust detection thresholds from the UI
- See defect pixels overlaid directly on the image
- Save the list of defect pixels for further analysis or production use

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/5b605817-72b9-4e89-b5f3-57a5008c557f" />

---

## Table of Contents

1. [What It Does & Why](#what-it-does--why)  
2. [Typical Use Cases](#typical-use-cases)  
3. [Project Structure](#project-structure)  
4. [Using the App Without Coding](#using-the-app-without-coding)  
   - [1. Getting an Image](#1-getting-an-image)  
   - [2. Setting Detection Parameters](#2-setting-detection-parameters)  
   - [3. Running Dark / Bright Search](#3-running-dark--bright-search)  
   - [4. Checking Results & Exporting CSV](#4-checking-results--exporting-csv)  
5. [How the Detection Works (High Level)](#how-the-detection-works-high-level)  
6. [Requirements & Installation](#requirements--installation)  
7. [How to Run](#how-to-run)  
8. [Notes & Limitations](#notes--limitations)

---

## What It Does & Why

Many machine-vision cameras and image sensors have:

- **Dark defects** – pixels that are consistently too dark
- **Bright defects** – pixels that are consistently too bright

Defect Pixel Finder helps you:

- Detect those pixels under controlled conditions (e.g. dark-field / flat-field images)
- Visualize them on top of the image
- Save them in a CSV for later processing or production calibration

The focus is on a **clear workflow** for engineers and operators:

> “Capture image → adjust thresholds → see defects → export CSV”

---

## Typical Use Cases

- **Sensor evaluation in R&D**
  - Check how many defect pixels a sensor has
  - Compare results between samples or production lots

- **Production / quality control**
  - Run a simple defect check before shipping cameras
  - Export defect lists to feed into your internal calibration tools

- **Field troubleshooting**
  - Inspect images from customer systems
  - Quickly see if strange specks or dots are due to fixed defect pixels

---

## Project Structure

```text
Defect_Pixel_Finder/
├─ main.py                  # Application entry point
├─ defect_tool_ui.py        # PyQt5 GUI (main widget)
├─ defect_detection.py      # Core defect detection algorithms (all Python/NumPy)
├─ draw_overlay.py          # Overlay drawing utilities
├─ core/
│  ├─ camera_controller.py  # eGrabber-based camera control (real or dummy)
│  ├─ camera_exceptions.py  # Camera-related exceptions
│  ├─ camera_facade.py      # Qt-friendly camera wrapper (CxpCamera)
│  └─ controller_pool.py    # Global registry of camera controllers
└─ workers/
   └─ grab_worker.py        # Background frame grabbing worker
```

- **UI**: `defect_tool_ui.py` + `draw_overlay.py`  
- **Algorithms**: `defect_detection.py` (pure Python + NumPy)  
- **Camera integration**: `core/` + `workers/grab_worker.py`  

---

## Using the App Without Coding

### 1. Getting an Image

In the main window you will find:

- **Connect** – search for and connect to the first available camera
- **Snap** – grab a single frame from the connected camera
- **Open Image** – load an image file from disk

Supported files (via OpenCV):

- `.tif` / `.tiff`
- `.png`
- `.jpg` / `.jpeg`
- `.bmp`

If you do **not** have a camera connected, you can just use **Open Image** and work
with saved images.

---

### 2. Setting Detection Parameters

In the side panel you can adjust:

- **Block size**
  - Size of the local neighborhood used to calculate an “average” value around each pixel
  - Smaller block:
    - More sensitive to local details
    - May detect more defects and more false positives  
  - Larger block:
    - Smoother background
    - Less sensitive to small local variations

- **Dark level (DN)**
  - How far below the local average a pixel must be to be considered “dark”
  - DN stands for “digital number” (intensity value)

- **Bright (±%)**
  - How far above (or below) the local average a pixel must be, in percent, to be considered “bright” or “too low”

- **PixelFormat (Load)**
  - When working with files, tells the app what kind of image it is, for example:
    - Mono8 / Mono16
    - Bayer RAW formats
    - BGR16

- **Histogram options (display only)**
  - Histogram normalize / stretch slider:
    - Only affect how the image **looks** on screen
    - Do not affect the detection result

---

### 3. Running Dark / Bright Search

After loading or capturing an image:

1. Adjust **Block size**, **Dark level (DN)**, and **Bright (±%)**.
2. Click **Dark Search** to find dark defect pixels.
3. Click **Bright Search** to find bright defect pixels.

The app will:

- Run the detection on the currently displayed image.
- Show defects as colored rectangles on the image.
- Update counters (how many dark / bright defects were found).

---

### 4. Checking Results & Exporting CSV

The main image view:

- Displays the image with overlays for each defect.
- Lets you visually confirm where the defects are:
  - Single isolated pixels
  - Small clusters
  - Specific regions (corners, edges, etc.)

When you are satisfied with the detection:

- Click **Export CSV** to save a defect list, for example:

```text
x,y
10,15
100,200
42,7
...
```

Each row represents one defect pixel at `(x, y)` (column, row).

You can then:

- Load this CSV into other tools (Python, Excel, internal test scripts).
- Use it as a defect mask in your production calibration process.

---

## How the Detection Works (High Level)

All detection logic is implemented in pure Python / NumPy in `defect_detection.py`.  
The core ideas:

1. **Normalize formats**
   - Convert different input formats (Bayer RAW, RGB, Mono) into a common
     grayscale representation suitable for analysis.

2. **Compute local averages**
   - Split the image into tiles based on **Block size**.
   - For each tile, compute an average intensity (local brightness level).
   - Interpolate between tile centers for smoother transitions.

3. **Compare each pixel to its local average**
   - For **dark defects**:
     - Pixels that are significantly lower than the local average are marked.
   - For **bright defects**:
     - Pixels that are significantly higher (or lower, based on your configuration) than the local average are marked.

4. **Group into regions**
   - The app can group neighboring defect pixels into clusters.
   - Rectangles around these clusters are drawn by `draw_overlay.py` over the image.

You do **not** need to understand the math to use the app, but it is all available
and readable for those who want to review or modify it.

---

## Requirements & Installation

### Python & OS

- **Python**: 3.8+ recommended  
- **OS**: Windows recommended for camera support (Coaxlink + eGrabber).  
  File-based mode can also work on other platforms if dependencies are available.

### Python dependencies

Minimal set:

- `PyQt5` – GUI (Qt)
- `numpy` – numeric operations
- `opencv-python` – image I/O and conversions
- `tifffile` – better TIFF support
- `egrabber` – Euresys camera SDK bindings (for live camera mode)

Example `requirements.txt`:

```text
PyQt5==5.15.11
numpy==1.24.4
opencv-python==4.11.0.86
tifffile==2023.7.10
egrabber==25.3.2.80
```

If you only want to work with image files and **not** with a camera, you can omit
`egrabber` (but camera features will be disabled).

---

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. From the project root, run:

   ```bash
   python main.py --log-level INFO --highdpi
   ```

   Options:

   - `--log-level` : `DEBUG`, `INFO`, `WARNING`, `ERROR`
   - `--highdpi`   : enable HiDPI scaling on high-resolution monitors
   - `--icon`      : optional window icon file

3. Use the GUI:

   - Connect a camera and press **Snap**, or press **Open Image**.
   - Adjust detection parameters.
   - Run **Dark Search** / **Bright Search**.
   - Inspect overlay and **Export CSV**.

---

## Notes & Limitations

- Designed for static, controlled images (e.g., flat-field / dark-field).
- If you run it on complex scenes (textured images, moving objects), many pixels
  may be marked as “defect” even though they are not sensor defects.
- Camera support currently targets Euresys Coaxlink + eGrabber; other SDKs would
  require adapting `core/camera_controller.py` and related code.

For most users, you can simply think of Defect Pixel Finder as:

> “A Python desktop app that lets you look for bad pixels,  
> without having to write a single line of code.”

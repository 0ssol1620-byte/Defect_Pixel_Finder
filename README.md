# Defect Pixel Finder

**Defect Pixel Finder** is a PyQt5-based desktop tool for detecting **dark and bright defect pixels**
in images from machine-vision cameras. 

The app is designed so that:

> Test / production engineers can inspect and export defect maps  
> **without writing any code**, using a simple GUI –  
> while developers still have a clean, testable Python implementation underneath.

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/5b605817-72b9-4e89-b5f3-57a5008c557f" />

---

## Table of Contents

1. [Key Concepts & Benefits](#key-concepts--benefits)  
2. [Typical Use Cases](#typical-use-cases)  
3. [High-level Architecture](#high-level-architecture)  
4. [UI Overview – Using the App Without Coding](#ui-overview--using-the-app-without-coding)  
   - [Connection & Image Acquisition](#connection--image-acquisition)  
   - [Defect Search Controls](#defect-search-controls)  
   - [Overlay & Statistics](#overlay--statistics)  
   - [CSV Export](#csv-export)  
5. [Detection Logic & Algorithms](#detection-logic--algorithms)  
6. [Requirements & Installation](#requirements--installation)  
7. [Running the Application](#running-the-application)  
8. [Limitations & Notes](#limitations--notes)

---

## Key Concepts & Benefits

### What Defect Pixel Finder does

- Connects to a CoaXPress camera (via Euresys eGrabber) **or** loads images from disk.
- Detects:
  - **Dark defects** (pixels darker than their local neighborhood)
  - **Bright defects** (pixels brighter than their local neighborhood)
- Supports:
  - Mono and Bayer RAW images
  - 8-bit and 10/12/14/16-bit formats
- Visualizes results as overlays on the image.
- Exports defects as a **CSV** defect map.

### Who it is for

- **Camera R&D / Image sensor engineers**
  - Characterize sensor defects during development.
- **Production / QA engineers**
  - Run regular defect maps on cameras before shipment.
- **FAE / Support**
  - Inspect customer images to see whether artifacts are due to pixel defects.

### Key benefits

- ✅ GUI-driven workflow; no programming required for normal use.  
- ✅ Algorithm behavior matches the legacy C implementation (thresholds, rounding, etc.).  
- ✅ Output CSV can be consumed by existing camera/sensor calibration pipelines.  
- ✅ Works with both live camera input and offline image files.

---

## Typical Use Cases

1. **Factory defect mapping**
   - Connect camera on a test bench.
   - Capture dark-field and bright-field frames.
   - Run Dark/Bright defect search.
   - Export CSV and feed it into production calibration process.

2. **Sensor evaluation**
   - Compare defect patterns across sensors or process lots.
   - Tune thresholds (DN / %) to match internal criteria.

3. **Field investigation**
   - Load customer-captured images.
   - Quickly see whether “hot” or “cold” pixels exist and where.

---

## High-level Architecture

```text
Defect_Pixel_Finder/
├─ main.py                  # Application entry point
├─ defect_tool_ui.py        # PyQt5 GUI (main widget)
├─ defect_detection.py      # Core defect detection algorithms
├─ draw_overlay.py          # Overlay rendering on images
├─ core/
│  ├─ camera_controller.py  # eGrabber-based camera control (real or dummy)
│  ├─ camera_exceptions.py  # Camera-related exceptions
│  ├─ camera_facade.py      # Qt-friendly CxpCamera wrapper
│  └─ controller_pool.py    # Global registry of CameraController instances
└─ workers/
   └─ grab_worker.py        # Background grabbing / draining worker
```

- **UI** layer: `defect_tool_ui.py` + `draw_overlay.py`  
- **Algorithms**: `defect_detection.py`  
- **Camera integration**: `core/` + `workers/grab_worker.py`  

---

## UI Overview – Using the App Without Coding

### Connection & Image Acquisition

In the main window (`DefectTool` from `defect_tool_ui.py`), you typically see:

- **Connect button**
  - Discovers cameras via `CxpCamera` / `CameraController`.
  - Reads camera `PixelFormat` to adjust valid DN ranges in the UI.

- **Snap button**
  - Grabs **one frame** from the camera.
  - Updates the display and clears previous overlays/statistics.

- **Open Image button**
  - Loads an image from disk (`.tif`, `.png`, `.jpg`, `.bmp`, etc.).
  - Uses OpenCV (`cv2.imread(..., IMREAD_UNCHANGED)`) to keep bit depth.
  - Lets you choose an assumed pixel format (`PixelFormat (Load)`) for analysis.

This means a non-programmer can:

1. Choose whether to work with a real camera or saved images.
2. Press a single button to get an image ready for defect inspection.

---

### Defect Search Controls

Key controls in the side panel:

- **Block size**
  - Tile size for computing local statistics (mean of neighborhood).
  - Smaller blocks → more locally sensitive, more potential false positives.  
  - Larger blocks → smoother background, more robust but less sensitive.

- **Dark level (DN)**
  - Absolute DN offset used for dark defect thresholds.
  - Internally combined with local averages for ≥10-bit images.

- **Bright (±%)**
  - Relative threshold (%) around local average.
  - Pixels deviating more than this percentage can be classified as bright or low.

- **PixelFormat (Load)**
  - When working with files, tells the app which format to assume:
    - Mono8 / Mono16
    - Bayer RAW variants (e.g. RGGB)
    - BGR16, etc.

- **Histogram normalize / stretch**
  - Affect only **display**, not the detection:
    - Normalize: auto-stretch histogram for better visual contrast.
    - Manual stretch slider: fine-tune brightness range for viewing.

To run detection:

1. Adjust `Block size`, `Dark level`, `Bright (±%)` to desired sensitivity.
2. Click **Dark Search** and/or **Bright Search**.
3. The overlay and counters update automatically.

---

### Overlay & Statistics

The main image view:

- Shows the current frame (live or loaded file).
- Overlays rectangles for detected:
  - Dark defects
  - Bright defects
- Uses `draw_overlay.py`:
  - `DefectLoc` objects store color, thickness, rect, optional label.
  - `DrawFigure` combines them and renders onto a copy of the image.

Statistics shown in the UI include:

- Count of dark and bright defects.
- Mean DN of the frame (or region).
- Optional cluster information, depending on configuration.

This allows you to visually verify:

- Whether defects are isolated or in clusters.
- Whether they appear mostly in certain regions (corners, center, etc.).

---

### CSV Export

The **Export CSV** button writes detected defects into a **C-style CSV**:

```text
:Vieworks Camera Defective Pixel Data
:H,:V
:dark field defective pixel
x,y
10,15
100,200
...
:Bright field defective pixel
x,y
42,7
...
```

- Dark defects first, then bright defects.
- Each line is `x,y` (column, row).
- The format is designed to be directly consumable by existing C/C++ tools or
  camera firmware scripts.

You can hand this CSV to other internal tools or embed it in your calibration pipeline.

---

## Detection Logic & Algorithms

The file `defect_detection.py` contains the core algorithms.

### Format Handling

- Converts disparate formats into a common working representation:
  - Bayer RAW → flat representation (`bayer_to_flat`)
  - RGB/BGR → green channel extraction (`extract_green_channel`)
  - BGR16 → Mono16 (scaled) to match original C logic
- Supports:
  - 8-bit (`uint8`)
  - ≥10-bit (`uint16` etc.) with appropriate scaling and thresholds

### Dark Defect Detection

Two main paths:

1. **Native (simpler) dark detection**
   - For Mono8: `pix >= thresholdDN`.
   - For ≥10-bit: `pix > local_avg + thresholdDN`.

2. **Cluster-based dark detection**
   - Uses block/tile averages plus thresholds to detect regions of abnormally dark pixels.
   - Aggregates into clusters and rectangular regions.

### Bright Defect Detection

- Uses block averages and **bilinear interpolation** (`avg_lin`) to estimate
  the expected value at each pixel.
- Compares each pixel against `(1 ± p/100) * avg_lin`.
- Pixels outside this band are tagged as bright (or low, depending on configuration).
- Clusters and rectangles are built from these binary masks.

### Histogram Normalization

- A display-only normalization (`histogram_normalize_c_identical`) reproduces the
  original C logic:
  - Same LUT behavior
  - Same integer rounding
- Ensures that images look familiar to users who are used to the legacy tool.

Overall, the algorithms are carefully implemented to **match the C reference** so
that threshold tuning and results remain consistent with existing workflows.

---

## Requirements & Installation

### Python & OS

- **Python**: 3.8+ recommended  
- **OS**: Windows is the main target (for camera SDK), but file-only mode can work elsewhere.

### Python dependencies

Core dependencies:

- `PyQt5` – GUI (Qt Widgets, Core, Gui)
- `numpy` – numeric operations and frame buffers
- `opencv-python` – image I/O and basic conversions
- `tifffile` – better TIFF support (16-bit, large images)
- `egrabber` – Euresys Coaxlink / eGrabber SDK (Python bindings) for live camera mode

Example `requirements.txt`:

```text
PyQt5==5.15.11
numpy==1.24.4
opencv-python==4.11.0.86
tifffile==2023.7.10
egrabber==25.3.2.80
```

> Without `egrabber` and a supported frame grabber, camera functions may fall back
> to dummy behavior, but **file-based defect inspection** will still work.

---

## Running the Application

From the project root:

```bash
pip install -r requirements.txt
python main.py --log-level INFO --highdpi
```

Arguments:

- `--log-level` – `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `WARNING`)
- `--highdpi` – enable Qt High-DPI scaling
- `--icon` – optional path to a custom icon file

### Typical workflow

1. **With camera**
   - Connect the camera to a Coaxlink board.
   - Start the app, click **Connect**, then **Snap**.
   - Adjust thresholds and run Dark/Bright search.
   - Inspect overlay, export CSV.

2. **With image files**
   - Start the app, click **Open Image**.
   - Make sure `PixelFormat (Load)` matches the file’s bit depth/layout.
   - Adjust parameters and run detection.
   - Export CSV for downstream use.

---

## Limitations & Notes

- Camera integration currently targets Euresys Coaxlink + eGrabber.
- Algorithms are designed for **static flat-field style images**; highly textured scenes
  are not ideal for defect detection and may yield many false positives.
- The tool does **not** attempt to model long-term drift or temperature behavior;
  it is focused on pixel-level defects in snapshots under controlled conditions.

If you need to extend the behavior (e.g., new CSV formats, alternative thresholds,
custom visualizations), you can modify `defect_detection.py`, `defect_tool_ui.py`, or
`draw_overlay.py` while keeping the rest of the structure intact.

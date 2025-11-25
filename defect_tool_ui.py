# -*- coding: utf-8 -*-
"""
defect_tool_ui.py (FINAL)
-------------------------
• Snap 모드: 사용자 'Snap' 버튼으로 1장씩 캡처 (자동 프리뷰 없음)
• PixelFormat 기반 Dark Level 범위(0~2^bits-1) & 기본값(100DN) 자동 설정
• Dark: data > avgTile + threshold(DN)  (네이티브 8U/16U 비교, 원본 C와 동일)
• Bright: 베이어면 BayerToFlat 후 검사 (임계=avg_lin±%, 스코어=avgTile 기준, 원본 C와 동일)
• Mono10/12/14/16: u16 정렬(right/left) 자동 판별 → 네이티브 DN 정규화/표시
• Mean (DN): 네이티브 DN 기준 평균(비트심도 표기)
• CSV: 헤더 없이 x,y (Dark → Bright 순)
• Download/Clear 버튼 제거, 검색 시 자동 덮어쓰기
• 모든 UI 요소 툴팁 포함 (Block size / Dark level / Bright% / 색상 등급)
"""

from __future__ import annotations
import contextlib
import sys
from typing import List, Tuple, Optional

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QFormLayout, QFrame, QSizePolicy, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect

# Camera facade
from core.camera_facade import CxpCamera

from defect_detection import (
    bayer_to_flat,
    bayer_flatfield_equalize,
    FindBrightFieldClusterRect_c_identical,   # ✅ 사용
    cluster_style,
    ClusterDataList, DefectPT, FindDarkFieldClusterRect_corr, histogram_normalize_c_identical
)


# Overlay (외부 모듈)
from draw_overlay import DefectLoc, DrawFigure


# ─────────────────────────────────────────────────────────────────────────────
# 다크 테마(QSS)
# ─────────────────────────────────────────────────────────────────────────────
QSS = """
*{font-family:'Inter','Segoe UI','Apple SD Gothic Neo','Malgun Gothic';}
QWidget{background:#0e1621;color:#e8eef4;font-size:13px;}
QGroupBox{
  border:1px solid #1e2a38;border-radius:10px;margin-top:16px;padding:10px 10px 10px 10px;
}
QGroupBox::title{
  subcontrol-origin: margin; left:12px; padding:0 6px; color:#8fb2ff; font-weight:600;
  background:transparent;
}
QLabel[role="title"]{color:#d8e6ff;font-weight:700;font-size:15px;}
QLabel[role="badge"]{
  background:#142235;border:1px solid #22344a;border-radius:12px;padding:4px 10px;
  color:#a9c6ff; font-weight:700;
}
QPushButton{
  background:#162231;border:1px solid #22344a;color:#d0deec;padding:8px 12px;border-radius:10px;
}
QPushButton:hover{ background:#19283b; }
QPushButton:pressed{ background:#0d1a28; }
QPushButton[accent="true"]{
  background:#2b5cff;border:1px solid #2b5cff;color:white;font-weight:700;
}
QPushButton[accent="true"]:hover{ background:#3b6cff; }
QPushButton[accent="true"]:pressed{ background:#224ce0; }
QComboBox, QSpinBox{
  background:#121c29;border:1px solid #22344a;color:#e8eef4;border-radius:8px;padding:6px 8px;
}
QCheckBox{ spacing:6px; }
QFrame#line{ background:#1e2a38; min-height:1px; max-height:1px;}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Bit-depth / DN 범위 유틸 + u16 정렬(좌/우) 자동 판별
# ─────────────────────────────────────────────────────────────────────────────
PIXELFORMAT_BITCANDIDATES = ("8", "10", "12", "14", "16")


# Bayer 픽셀 포맷 문자열에서 패턴 추출 ('BayerRG12' → 'RGGB' 가정)
def _infer_bayer_pattern_from_pixfmt(pixfmt: str) -> str:
    s = (pixfmt or "").upper()
    if "BAYER" in s and "RG" in s: return "RGGB"
    if "BAYER" in s and "GR" in s: return "GRBG"
    if "BAYER" in s and "GB" in s: return "GBRG"
    if "BAYER" in s and "BG" in s: return "BGGR"
    return "RGGB"


def _infer_frame_bit_depth_strict(frame: np.ndarray) -> int:
    """
    카메라 PixelFormat이 없을 때, 프레임의 최대 DN으로 bit-depth(8/10/12/14/16)를 추정.
    u8→8, u16→vmax 기준 휴리스틱.
    """
    if frame is None:
        return 16
    if frame.dtype == np.uint8:
        return 8
    if frame.dtype != np.uint16:
        return 16

    # 컬러라도 최대 DN은 모든 채널에서 동일 스케일이라고 가정
    vmax = int(frame.max())
    if   vmax <=  1023:  return 10
    elif vmax <=  4095:  return 12
    elif vmax <= 16383:  return 14
    else:                return 16

# ─────────────────────────────────────────────────────────────────────────────
# Euresys(eGrabber) Bayer→Color 변환 어댑터 (1st try), OpenCV는 fallback
# ─────────────────────────────────────────────────────────────────────────────
def _bayer_to_bgr8_euresys_first(raw2d: np.ndarray, pixfmt: str, *, src_bits: int) -> np.ndarray:
    """
    Euresys eGrabber의 imageConvert를 우선 사용해 Bayer→BGR8 변환.
    실패 시 내부 demosaic(bayer_to_flat)로 폴백.
    """
    if raw2d is None or raw2d.ndim != 2:
        raise ValueError("raw2d must be a single-channel Bayer image")

    # 1) Euresys 시도 (generated.cEGrabber 직접 호출)
    try:
        import ctypes as ct
        from egrabber.generated import cEGrabber as eur

        def _pf_val(name: str) -> int:
            return eur.Eur_EGenTL_imageGetPixelFormatValue(name.encode("ascii"))

        pf = (pixfmt or "").strip()
        if not pf or "BAYER" not in pf.upper():
            pf = "BayerRG8" if raw2d.dtype == np.uint8 else "BayerRG16"

        src_pf_val = _pf_val(pf)
        dst_pf_val = _pf_val("BGR8")

        h, w = raw2d.shape
        src_stride = raw2d.strides[0]          # bytes per line
        dst_stride = w * 3
        dst_buf = np.empty((h, w, 3), dtype=np.uint8)

        class ImageConvertInput(ct.Structure):
            _fields_ = [
                ("width", ct.c_size_t), ("height", ct.c_size_t),
                ("pixelFormat", ct.c_uint64),
                ("data", ct.c_void_p), ("stride", ct.c_size_t),
            ]
        class ImageConvertOutput(ct.Structure):
            _fields_ = [
                ("pixelFormat", ct.c_uint64),
                ("data", ct.c_void_p), ("stride", ct.c_size_t),
                ("bufferSize", ct.c_size_t),
            ]

        inp = ImageConvertInput(
            width=w, height=h, pixelFormat=ct.c_uint64(src_pf_val),
            data=raw2d.ctypes.data_as(ct.c_void_p), stride=src_stride
        )
        out = ImageConvertOutput(
            pixelFormat=ct.c_uint64(dst_pf_val),
            data=dst_buf.ctypes.data_as(ct.c_void_p), stride=dst_stride,
            bufferSize=dst_buf.nbytes
        )

        eur.Eur_EGenTL_imageConvert(ct.byref(inp), ct.byref(out))
        return dst_buf

    except Exception:
        # 2) 폴백: 내부 demosaic
        pat = _infer_bayer_pattern_from_pixfmt(pixfmt)
        if raw2d.dtype == np.uint16 and src_bits >= 10:
            max_native = float((1 << int(src_bits)) - 1)
            bgr16 = bayer_to_flat(raw2d, pattern=pat)  # uint16 BGR
            bgr8 = np.clip(bgr16.astype(np.float32) * (255.0 / max_native), 0, 255).astype(np.uint8)
            return bgr8
        else:
            return bayer_to_flat(raw2d, pattern=pat).astype(np.uint8)

def infer_bit_depth_from_pixfmt(pixfmt: Optional[str]) -> int:
    s = (pixfmt or "").upper()
    for k in PIXELFORMAT_BITCANDIDATES:
        if k in s:
            return int(k)
    return 16

def infer_bit_depth_from_frame(frame: Optional[np.ndarray]) -> int:
    if frame is None:
        return 16
    if frame.dtype == np.uint8:
        return 8
    if frame.dtype == np.uint16:
        return 16
    return 16

def pick_effective_bit_depth(pixfmt: Optional[str], frame: Optional[np.ndarray]) -> int:
    b = infer_bit_depth_from_pixfmt(pixfmt)
    if b in (8,10,12,14,16):
        return b
    return infer_bit_depth_from_frame(frame)

def set_dark_spin_range_by_frame(self, frame: np.ndarray) -> None:
    """
    카메라 PixelFormat 없이도, 로드한 프레임으로 Dark Level 범위를 자동 설정.
    uint8 → 8bit, uint16 → 최대 DN으로 10/12/14/16bit 추정.
    """
    if frame is None:
        return
    if frame.dtype == np.uint8:
        bits = 8
    elif frame.dtype == np.uint16:
        vmax = int(frame.max())
        # 휴리스틱: 최대 DN으로 비트심도 추정
        if   vmax <= 1023:   bits = 10
        elif vmax <= 4095:   bits = 12
        elif vmax <= 16383:  bits = 14
        else:                bits = 16
    else:
        bits = 16  # 보수적 기본값

    max_dn = (1 << bits) - 1
    self.spnDark.setRange(0, max_dn)
    self.spnDark.setValue(min(100, max_dn))
    self.spnDark.setToolTip(
        "Dark Level (DN)\n"
        "• 다크 검사에서 같은 블록 평균에 더해지는 절대 임계치입니다.\n"
        "• 판정식: data > avgTile + DarkLevel\n"
        "• 값이 작을수록 민감(오검↑), 값이 클수록 보수적(미검↑).\n"
        f"• 현재 프레임 기반 범위: 0 ~ {max_dn} DN (추정 bit-depth: {bits}-bit)"
    )

def _awb_grayworld(bgr8: np.ndarray) -> np.ndarray:
    # 채널 평균이 같아지도록 R/B gain을 맞춘다(G는 기준).
    b, g, r = cv2.split(bgr8.astype(np.float32))
    gb = g.mean() + 1e-6
    r_gain = gb / (r.mean() + 1e-6)
    b_gain = gb / (b.mean() + 1e-6)
    r = np.clip(r * r_gain, 0, 255)
    b = np.clip(b * b_gain, 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

def _apply_gamma(bgr8: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    inv = 1.0 / max(1e-6, gamma)
    lut = (np.linspace(0, 1, 256) ** inv) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr8, lut)


# ---- Mono10/12/14/16 u16 정렬/스케일 유틸 ----
def _detect_u16_alignment(gray16: np.ndarray, src_bits: int) -> str:
    """
    u16 그레이가 src_bits 유효비트를 어디에 정렬했는지 추정 ('right'/'left').
    휴리스틱: 값 범위 + 하위 shift비트 0 비율.
    """
    if gray16.dtype != np.uint16:
        return 'right'
    bits = int(src_bits)
    shift = max(0, 16 - bits)
    max_native = (1 << bits) - 1
    vmax = int(gray16.max())
    # 범위 기반
    if vmax <= max_native + 16:
        return 'right'
    if vmax >= (max_native << shift) - 64:
        return 'left'
    # 하위 비트 0 비율
    if shift > 0:
        lower_mask = (1 << shift) - 1
        zeros = float(((gray16 & lower_mask) == 0).mean())
        return 'left' if zeros > 0.9 else 'right'
    return 'right'

def u16_to_native_dn(gray16: np.ndarray, src_bits: int) -> np.ndarray:
    """
    u16 그레이를 '네이티브 DN(0..2^bits-1)'로 정규화해서 u16로 반환.
    right → 그대로, left → >> (16-bits)
    """
    assert gray16.dtype == np.uint16
    align = _detect_u16_alignment(gray16, src_bits)
    shift = max(0, 16 - int(src_bits))
    if align == 'left' and shift > 0:
        return (gray16 >> shift).astype(np.uint16)
    return gray16

def u16_to_u8_display(gray16: np.ndarray, src_bits: int) -> np.ndarray:
    """
    표시용 8U로 변환. (네이티브 DN → 8U 스케일)
    """
    native = u16_to_native_dn(gray16, src_bits)  # 0..max_native
    max_native = float((1 << int(src_bits)) - 1)
    if max_native <= 0:
        return np.zeros_like(gray16, dtype=np.uint8)
    return np.clip(native * (255.0 / max_native), 0, 255).astype(np.uint8)

def set_dark_spin_range_by_pixfmt(spin_widget, pixfmt: Optional[str], default_dn: int = 100) -> int:
    """
    PixelFormat 문자열에서 비트심도를 추출해 Dark Level 스핀박스의 범위/기본값/툴팁을 설정한다.
    반환값: 추정된 bit-depth (8/10/12/14/16)

    예) 'Mono12', 'Mono12p', 'BayerRG10', 'RGB16' → 12/12/10/16
    인식 실패 시 16으로 보수적 설정.
    """
    s = (pixfmt or "").strip().upper()
    # 가장 흔한 패턴: 숫자 포함 여부로 추출 (Mono12, BayerRG10, RGB16, Mono12P 등)
    bits = 16
    for k in ("8", "10", "12", "14", "16"):
        if k in s:
            bits = int(k)
            break

    max_dn = (1 << bits) - 1
    spin_widget.setRange(0, max_dn)
    spin_widget.setValue(min(int(default_dn), max_dn))
    spin_widget.setToolTip(
        "Dark Level (DN)\n"
        "• 다크(암야) 검사에서 같은 블록 평균(avgTile)에 더해지는 절대 임계치입니다.\n"
        "• 판정식: data > avgTile + DarkLevel\n"
        "• 값이 작을수록 민감(오검↑), 값이 클수록 보수적(미검↑).\n"
        f"• 현재 PixelFormat({pixfmt or 'N/A'}) 기준 범위: 0 ~ {max_dn} DN ({bits}-bit)"
    )
    return bits


class GridView(QLabel):
    mouseHover = pyqtSignal(int, int)
    selectionMade = pyqtSignal(int, int, int, int)  # x, y, w, h (image coords)
    _DEF_ZOOM_STEP = 2.0
    _DEF_ZOOM_MIN  = 0.05
    _DEF_ZOOM_MAX  = 1024.0

    def __init__(self):
        super().__init__("No Signal")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(900, 600)

        self._bgr: Optional[np.ndarray] = None
        self._overlay = None
        self._overlay_rects: list = []

        from PyQt5.QtCore import QPointF
        self._zoom: float = 1.0
        self._fit_mode: bool = True
        self._pan = QPointF(0.0, 0.0)
        self._panning: bool = False
        self._last_pos = None

        # 이미지가 화면보다 작을 때도 살짝 움직일 수 있는 여유(픽셀)
        self._overscroll: float = 120.0

        # ★ Erase(선택) 상태
        self._erase_enabled: bool = False
        self._sel_dragging: bool = False
        self._sel_start_img = None  # (ix, iy)
        self._sel_end_img = None  # (ix, iy)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_erase_enabled(self, enabled: bool) -> None:
        """Erase 모드 on/off."""
        self._erase_enabled = bool(enabled)
        self._sel_dragging = False
        self._sel_start_img = None
        self._sel_end_img = None
        self.setCursor(Qt.ArrowCursor)
        self.update();
        self.repaint()

    def _current_selection_rect_img(self) -> Optional[QRect]:
        """이미지 좌표계의 선택 사각형(Rect) 반환. 없으면 None."""
        if self._sel_start_img is None or self._sel_end_img is None:
            return None
        x0, y0 = self._sel_start_img
        x1, y1 = self._sel_end_img
        l = min(x0, x1)
        t = min(y0, y1)
        r = max(x0, x1)
        b = max(y0, y1)
        if r <= l or b <= t:
            return None
        # +1: 마지막 픽셀까지 포함 (inclusive) → 이후 로직에서 폭/높이로 쓰기 쉬움
        return QRect(l, t, (r - l + 1), (b - t + 1))

    # ---------- Public ----------
    def set_frame(self, bgr: np.ndarray, overlay) -> None:
        self._bgr = bgr
        self._overlay = overlay
        self._fit_mode = True
        from PyQt5.QtCore import QPointF
        self._pan = QPointF(0.0, 0.0)
        self.update(); self.repaint()

    def set_overlay_rects(self, rects: list) -> None:
        self._overlay_rects = rects or []
        self.update(); self.repaint()

    # ---------- Internals ----------
    def _image_size(self) -> tuple[int, int]:
        if self._bgr is None:
            return 0, 0
        h, w = self._bgr.shape[:2]
        return w, h

    def _fit_zoom(self) -> float:
        w, h = self._image_size()
        if w == 0 or h == 0:
            return 1.0
        vw = max(1, self.width())
        vh = max(1, self.height())
        z = min(vw / float(w), vh / float(h))
        return max(globals().get("ZOOM_MIN", self._DEF_ZOOM_MIN),
                   min(z, globals().get("ZOOM_MAX", self._DEF_ZOOM_MAX)))

    def _scale_effective(self) -> float:
        if self._fit_mode:
            return self._fit_zoom()
        return max(globals().get("ZOOM_MIN", self._DEF_ZOOM_MIN),
                   min(float(self._zoom), globals().get("ZOOM_MAX", self._DEF_ZOOM_MAX)))

    def _content_rect(self) -> tuple[float, float, float, float, float]:
        w, h = self._image_size()
        z = self._scale_effective()
        dw, dh = (w * z), (h * z)
        ox = (self.width() - dw) / 2.0 + self._pan.x()
        oy = (self.height() - dh) / 2.0 + self._pan.y()
        return ox, oy, dw, dh, z

    def _clamp_pan(self) -> None:
        """
        pan_x, pan_y는 '센터 기준 추가 이동' 값.
        ox = (W - dw)/2 + pan_x,  xL = ox, xR = ox + dw.
        제약: xL <= +m  &&  xR >= W - m  (m = overscroll)  → 가장자리에 m 픽셀 여유.
        같은 방식으로 y도 처리.
        """
        w_img, h_img = self._image_size()
        if w_img == 0 or h_img == 0:
            return
        Wv, Hv = self.width(), self.height()
        _, _, dw, dh, _ = self._content_rect()
        m = float(getattr(self, "_overscroll", 0.0))

        # 공통 보조값
        base_x = (Wv - dw) / 2.0   # pan=0일 때 좌상단 x
        base_y = (Hv - dh) / 2.0

        from PyQt5.QtCore import QPointF

        # 수평
        if dw <= Wv:
            # 화면보다 작으면 중앙 고정 (+/- m로 살짝 움직이게 하려면 아래 주석 해제)
            px_min, px_max = -m, +m
        else:
            # 제약식: base_x + pan_x <= +m  → pan_x <= m - base_x
            #        base_x + pan_x + dw >= Wv - m  → pan_x >= Wv - m - dw - base_x
            px_min = (Wv - m - dw) - base_x
            px_max = (m) - base_x

        # 수직
        if dh <= Hv:
            py_min, py_max = -m, +m
        else:
            py_min = (Hv - m - dh) - base_y
            py_max = (m) - base_y

        px = max(px_min, min(px_max, self._pan.x()))
        py = max(py_min, min(py_max, self._pan.y()))
        self._pan = QPointF(px, py)

    def _map_widget_to_image(self, pos) -> Optional[tuple[int, int]]:
        """위젯 좌표 → 이미지 정수 좌표(클램프). 이미지 밖이면 가장 가까운 경계로 스냅."""
        if self._bgr is None:
            return None
        ox, oy, dw, dh, z = self._content_rect()
        # 이미지 기준 좌표(부동소수점)
        x = (pos.x() - ox)
        y = (pos.y() - oy)
        if dw <= 0 or dh <= 0 or z <= 0:
            return None
        # 이미지 밖이면 가장 가까운 가장자리로 클램프 → 가장자리에서도 커서 고정 줌 가능
        x = float(np.clip(x, 0.0, max(0.0, dw - 1.0)))
        y = float(np.clip(y, 0.0, max(0.0, dh - 1.0)))

        ix = int(x / z)
        iy = int(y / z)
        w, h = self._image_size()
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        return ix, iy

    # ---------- Events ----------
    def wheelEvent(self, ev):
        if self._bgr is None:
            return
        cursor = ev.pos()
        before = self._map_widget_to_image(cursor)
        if before is None:
            return

        delta = ev.angleDelta().y()
        if delta == 0:
            return

        if self._fit_mode:
            self._fit_mode = False
            self._zoom = self._fit_zoom()

        step = globals().get("ZOOM_STEP", self._DEF_ZOOM_STEP)
        step = step if delta > 0 else (1.0 / step)
        new_zoom = max(globals().get("ZOOM_MIN", self._DEF_ZOOM_MIN),
                       min(self._zoom * step, globals().get("ZOOM_MAX", self._DEF_ZOOM_MAX)))

        ix, iy = before
        self._zoom = new_zoom

        # 커서 고정 줌: pan' = cursor - [center_offset + scaled_image_point]
        w, h = self._image_size()
        dw, dh = (w * new_zoom), (h * new_zoom)
        cx = (self.width() - dw) / 2.0
        cy = (self.height() - dh) / 2.0

        from PyQt5.QtCore import QPointF
        self._pan = QPointF(
            cursor.x() - (cx + ix * new_zoom),
            cursor.y() - (cy + iy * new_zoom)
        )
        self._clamp_pan()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self._bgr is None:
            return
        self._fit_mode = not self._fit_mode
        if self._fit_mode:
            from PyQt5.QtCore import QPointF
            self._pan = QPointF(0.0, 0.0)
        else:
            self._zoom = self._fit_zoom()
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            from PyQt5.QtCore import QPointF
            self._pan = QPointF(0.0, 0.0)
        else:
            self._clamp_pan()
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._bgr is not None:
            if self._erase_enabled:
                m = self._map_widget_to_image(ev.pos())
                if m is not None:
                    self._sel_dragging = True
                    self._sel_start_img = m
                    self._sel_end_img = m
                    self.setCursor(Qt.CrossCursor)
                    self.update()
                return  # Erase 모드에서는 패닝 비활성
            # 기존 패닝
            self._panning = True
            self._last_pos = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._erase_enabled and self._sel_dragging:
            m = self._map_widget_to_image(ev.pos())
            if m is not None:
                self._sel_end_img = m
                self.update()
            # hover 업데이트
            m2 = self._map_widget_to_image(ev.pos())
            if m2 is None:
                self.mouseHover.emit(-1, -1)
            else:
                self.mouseHover.emit(m2[0], m2[1])
            return

        # 기존 패닝 경로
        if self._panning and self._last_pos is not None:
            d = ev.pos() - self._last_pos
            if self._fit_mode:
                self._fit_mode = False
                self._zoom = self._fit_zoom()
            from PyQt5.QtCore import QPointF
            self._pan += QPointF(float(d.x()), float(d.y()))
            self._last_pos = ev.pos()
            self._clamp_pan()
            self.update()

        m = self._map_widget_to_image(ev.pos())
        if m is None:
            self.mouseHover.emit(-1, -1)
        else:
            self.mouseHover.emit(m[0], m[1])
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if self._erase_enabled and self._sel_dragging:
                self._sel_dragging = False
                self.setCursor(Qt.ArrowCursor)
                rect = self._current_selection_rect_img()
                # 선택이 유효하면 알림(이미지 좌표계)
                if rect is not None:
                    self.selectionMade.emit(rect.x(), rect.y(), rect.width(), rect.height())
                # 선택 시각화 초기화
                self._sel_start_img = None
                self._sel_end_img = None
                self.update()
                return
            # 기존 패닝 종료
            self._panning = False
            self._last_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0e1621"))

        # 배경 그리드
        p.setPen(QPen(QColor(30, 42, 56), 1))
        step = 32
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)

        if self._bgr is None:
            p.end()
            return

        # 1) DrawFigure 합성
        disp = self._bgr
        if self._overlay is not None and hasattr(self._overlay, "render_on"):
            try:
                tmp = self._overlay.render_on(self._bgr, alpha=0.55, draw_centers=False)
                if isinstance(tmp, np.ndarray) and tmp.ndim == 3 and tmp.shape[2] >= 3:
                    if tmp.dtype != np.uint8:
                        tmp = np.clip(tmp, 0, 255).astype(np.uint8)
                    disp = tmp[..., :3]
            except Exception:
                pass

        # 2) QImage로 그리기
        rgb = disp[..., ::-1].copy()
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.setRenderHint(QPainter.Antialiasing, False)
        ox, oy, dw, dh, z = self._content_rect()
        p.save()
        p.translate(int(round(ox)), int(round(oy)))
        p.scale(z, z)
        p.drawImage(0, 0, qimg)

        # 3) UI 사각형(= 디펙 표시)
        if self._overlay_rects:
            for (x, y, ww, hh, bgr, thick) in self._overlay_rects:
                pen = QPen(QColor(int(bgr[2]), int(bgr[1]), int(bgr[0])))
                pen.setWidth(1 if (int(ww) == 1 and int(hh) == 1) else max(1, min(int(thick), 2)))
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(int(x), int(y), int(ww), int(hh))
        p.restore()

        # 4) (추가) 선택 영역(Erase) 표시: 위젯 좌표계 기준 반투명 박스
        if self._erase_enabled and self._sel_start_img is not None and self._sel_end_img is not None:
            # 이미지 좌표 → 위젯 좌표
            x0, y0 = self._sel_start_img
            x1, y1 = self._sel_end_img
            l = min(x0, x1);
            t = min(y0, y1)
            r = max(x0, x1);
            b = max(y0, y1)
            # 픽셀 경계 보이게 우하단 +1
            wl = int(round(ox + l * z))
            wt = int(round(oy + t * z))
            wr = int(round(ox + (r + 1) * z))
            wb = int(round(oy + (b + 1) * z))
            sel_rect = QRect(wl, wt, max(1, wr - wl), max(1, wb - wt))

            p.setPen(QPen(QColor(180, 220, 255, 220), 1, Qt.DashLine))
            p.setBrush(QColor(120, 170, 255, 60))
            p.drawRect(sel_rect)

        p.end()


class Legend(QWidget):
    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        entries = [("Blue","<15"),("Green","<25"),("Orange","<30"),("Red","≥30")]
        colors = [(0,0,255),(0,180,0),(180,80,0),(255,0,0)]
        tips = [
            "Blue: diffsum < 15 (경미)",
            "Green: diffsum < 25 (중간)",
            "Orange: diffsum < 30 (높음)",
            "Red: diffsum ≥ 30 (심각)"
        ]
        for (name, rng), c, tip in zip(entries, colors, tips):
            sw = QLabel(); sw.setFixedSize(14,14)
            sw.setStyleSheet(f"background: rgb({c[0]},{c[1]},{c[2]}); border-radius:3px;")
            txt = QLabel(f"{name} {rng}"); txt.setStyleSheet("color:#9db7d7;")
            txt.setToolTip(
                "색상 등급\n"
                "• diffsum = 클러스터의 픽셀 스코어 합\n"
                "• Bright 스코어: |100 - val*100/avgTile| (% 편차)\n"
                "• Dark 스코어:  ((val - avgTile)*100) / DarkLevel\n"
                f"→ {tip}"
            )
            box = QHBoxLayout(); w = QWidget(); w.setLayout(box)
            box.setContentsMargins(0,0,0,0); box.setSpacing(6)
            box.addWidget(sw); box.addWidget(txt)
            lay.addWidget(w)
        lay.addStretch(1)
        self.setToolTip(
            "색상 범례\n"
            "• Blue/Green/Orange/Red 는 디펙 클러스터의 심각도 등급을 의미합니다.\n"
            "• diffsum 기준: <15, <25, <30, ≥30"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 메인 툴 UI
# ─────────────────────────────────────────────────────────────────────────────
class DefectTool(QWidget):
    def __init__(self):
        super().__init__()
        # ───────────────── Window / Theme ─────────────────
        self.setWindowTitle("Defect Pixel Finder")
        self.setStyleSheet(QSS)
        self.setToolTip(
            "Defect Pixel Finder\n"
            "• Snap으로 1장씩 촬영하거나 Open Image로 파일을 열어 Dark/Bright 검사를 수행합니다.\n"
        )

        # ───────────────── Camera / Viewer ─────────────────
        self.cam = CxpCamera()
        self.view = GridView()

        # ───────────────── Controls (left panel) ─────────────────
        # Block size
        self.cmbBlock = QComboBox()
        for k in (4, 8, 16, 32, 64):
            self.cmbBlock.addItem(str(k))
        self.cmbBlock.setCurrentText("32")
        self.cmbBlock.setToolTip(
            "Block size (블록 크기)\n"
            "• 이미지를 block_size×block_size 타일로 나눠 지역 평균을 계산합니다.\n"
            "• 작을수록 미세 결함에 민감(오검↑), 클수록 부드러운 배경(미세 결함 둔감)."
        )

        # Dark level (DN)
        self.spnDark = QSpinBox()
        self.spnDark.setRange(0, 65535)
        self.spnDark.setValue(100)
        self.spnDark.setToolTip(
            "Dark Level (DN)\n"
            "• 다크 검사에서 같은 블록 평균에 더해지는 절대 임계치입니다.\n"
            "• 판정식: data > avgTile + DarkLevel"
        )

        # Bright ±%
        self.spnBrightPct = QSpinBox()
        self.spnBrightPct.setRange(1, 99)
        self.spnBrightPct.setValue(15)
        self.spnBrightPct.setToolTip(
            "Bright (±%)\n"
            "• 브라이트 검사에서 지역 기준(avg_lin) 대비 허용하는 상대 편차입니다.\n"
            "• 판정식: val > avg_lin×(1+p/100) 또는 val < avg_lin×(1-p/100)"
        )

        # Overlay toggle
        self.chkOverlay = QCheckBox("Show overlay")
        self.chkOverlay.setChecked(True)
        self.chkOverlay.setToolTip("오버레이(디펙 표시) On/Off")

        # Histogram normalize(표시 전용, C동일 Normalize)
        self.chkHistNorm = QCheckBox("Histogram normalize (display)")
        self.chkHistNorm.setToolTip(
            "표시용 전역 히스토그램 평활화(HE: CDF 기반).\n"
            "• 원본 C Normalize와 동일 의미(Cut 없음, nStartVal=0).\n"
            "• Stretch %는 Normalize가 꺼져 있을 때만 적용됩니다."
        )

        # Stretch % (Normalize가 꺼져 있을 때만 의미)
        self.sldNormStrength = QSlider(Qt.Horizontal)
        self.sldNormStrength.setRange(1, 100)  # 1~100%
        self.sldNormStrength.setValue(100)  # 100% = 무보정(원본)
        self.sldNormStrength.setTracking(True)
        self.sldNormStrength.setToolTip(
            "Stretch Percent (C와 동일)\n"
            "• 100%: 무보정(원본)\n"
            "• 90% : 하·상 5% 클리핑 후 0~Max로 스트레치\n"
            "• 50% : 하·상 25% 클리핑 후 0~Max로 스트레치\n"
            "※ Normalize가 꺼져 있을 때만 적용됩니다."
        )

        # Buttons
        self.btnConnect = QPushButton("Connect");
        self.btnConnect.setProperty("accent", True)
        self.btnConnect.setToolTip("카메라에 연결합니다. 연결되면 'Snap'으로 한 장씩 캡처하세요.")
        self.btnSnap = QPushButton("Snap");
        self.btnSnap.setProperty("accent", True)
        self.btnSnap.setToolTip("1장을 캡처합니다. 캡처 직후 이전 오버레이/카운트를 초기화합니다.")
        self.btnOpen = QPushButton("Open Image")
        self.btnOpen.setToolTip("파일에서 이미지를 불러옵니다. (카메라 없이도 검사 가능)")

        self.btnDark = QPushButton("Dark Search");
        self.btnDark.setToolTip(
            "Dark 검사 실행 — data > avgTile + DarkLevel (DN).")
        self.btnBright = QPushButton("Bright Search");
        self.btnBright.setToolTip(
            "Bright 검사 실행 — 베이어 RAW는 BayerToFlat(4위상 평탄화) 또는 RGB 평탄화 후 G 채널 기준.\n"
            "블록 평균 + 인접 기울기 보간(dAvg) 대비 ±p%."
        )

        self.btnSaveCsv = QPushButton("Export CSV")
        self.btnSaveCsv.setToolTip("검출 결과를 원본 C 스타일 CSV로 저장합니다.")

        # Counters/Mean badges
        self.lblDarkCnt = QLabel("Dark: 0");
        self.lblDarkCnt.setProperty("role", "badge")
        self.lblDarkCnt.setToolTip("검출된 Dark(HOT) 픽셀 수")
        self.lblBrightCnt = QLabel("Bright: 0");
        self.lblBrightCnt.setProperty("role", "badge")
        self.lblBrightCnt.setToolTip("검출된 Bright(Dead) 픽셀 수")
        self.lblMean = QLabel("Mean (DN): -");
        self.lblMean.setProperty("role", "badge")
        self.lblMean.setToolTip("현재 프레임의 평균 밝기 (네이티브 DN)")

        # ───────────────── Groups ─────────────────
        # Acquisition
        gAcq = QGroupBox("Acquisition");
        gAcq.setToolTip("카메라 연결/촬영 및 파일 로드")
        fAcq = QFormLayout()
        fAcq.addRow(self.btnConnect)
        fAcq.addRow(self.btnSnap)
        fAcq.addRow(self.btnOpen)
        gAcq.setLayout(fAcq)

        # Detection
        gDetect = QGroupBox("Detection");
        gDetect.setToolTip("검사 조건 설정")
        fDet = QFormLayout()
        fDet.addRow(QLabel("Block size"), self.cmbBlock)
        fDet.addRow(QLabel("Dark Level (DN)"), self.spnDark)
        fDet.addRow(QLabel("Bright ±%"), self.spnBrightPct)
        gDetect.setLayout(fDet)

        # Actions
        gActions = QGroupBox("Actions");
        gActions.setToolTip("표시 옵션 / 톤 보정")
        la = QVBoxLayout()
        row1 = QHBoxLayout();
        row1.addWidget(self.btnDark);
        row1.addWidget(self.btnBright);
        la.addLayout(row1)
        la.addWidget(self.chkOverlay)
        la.addWidget(self.chkHistNorm)

        self.chkErase = QCheckBox("Erase (drag to exclude)")
        self.chkErase.setToolTip("디펙 검출 후, 마우스로 드래그한 영역의 디펙 픽셀을 제거합니다. CSV에서도 제외됩니다.")
        la.addWidget(self.chkErase)
        # (Stretch % 슬라이더는 숨김 기본; 필요 시 UI에 노출하려면 아래 주석 해제)
        # formTone = QFormLayout(); formTone.addRow("Stretch %", self.sldNormStrength); la.addLayout(formTone)
        gActions.setLayout(la)

        # Export
        gExport = QGroupBox("Export");
        gExport.setToolTip("검출 결과 저장")
        le = QVBoxLayout();
        le.addWidget(self.btnSaveCsv);
        gExport.setLayout(le)

        # Left panel layout
        badge = QHBoxLayout()
        badge.addWidget(self.lblDarkCnt);
        badge.addSpacing(8)
        badge.addWidget(self.lblBrightCnt);
        badge.addSpacing(8)
        badge.addWidget(self.lblMean);
        badge.addStretch(1)

        left = QVBoxLayout()
        t = QLabel("Defect Pixel Finder");
        t.setProperty("role", "title")
        t.setToolTip("Defect Pixel Finder – 원본 C와 의미 일치하는 검사/표시")
        left.addWidget(t)
        left.addLayout(badge)
        left.addWidget(gAcq)
        left.addWidget(gDetect)
        left.addWidget(gActions)
        left.addWidget(gExport)
        left.addStretch(1)
        panel = QWidget();
        panel.setLayout(left);
        panel.setFixedWidth(390)

        # Right: Legend + Preview + Hover DN
        legend = Legend()
        right = QVBoxLayout()
        right.addWidget(legend)
        self.lblHover = QLabel("x:- y:-  DN:-");
        self.lblHover.setStyleSheet("color:#9db7d7;")
        right.addWidget(self.lblHover)
        line = QFrame();
        line.setObjectName("line");
        right.addWidget(line)
        right.addWidget(self.view, 1)
        viewer = QWidget();
        viewer.setLayout(right)

        sp = QSplitter();
        sp.addWidget(panel);
        sp.addWidget(viewer);
        sp.setStretchFactor(1, 1)
        root = QVBoxLayout(self);
        root.addWidget(sp)
        self._latest_dark = {"pts": [], "clusters": None, "frame_size": None}  # (w, h)
        self._latest_bright = {"pts": [], "clusters": None, "frame_size": None}  # (w, h)
        # ───────────────── State ─────────────────
        self._src_frame: Optional[np.ndarray] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_bgr: Optional[np.ndarray] = None
        self._frame_source = "camera"
        self._assumed_load_pixfmt = ""

        self._dark_pts: List[Tuple[int, int]] = []
        self._bright_pts: List[Tuple[int, int]] = []
        self._overlay = DrawFigure()
        self._overlay_rects: List[Tuple[int, int, int, int, Tuple[int, int, int], int]] = []

        self._dark_clusters: Optional[ClusterDataList] = None
        self._bright_clusters: Optional[ClusterDataList] = None

        # ───────────────── Signals ─────────────────
        self.cam.new_frame.connect(self.on_frame)
        self.btnConnect.clicked.connect(self.on_connect)
        self.btnSnap.clicked.connect(self.on_snap)
        self.btnOpen.clicked.connect(self.on_open_image)
        self.btnDark.clicked.connect(self.on_dark_search)
        self.btnBright.clicked.connect(self.on_bright_search)
        self.btnSaveCsv.clicked.connect(self.on_save_csv)

        self.view.mouseHover.connect(self.on_hover)
        self.chkOverlay.toggled.connect(lambda _checked: self.refresh_view())
        # 기존 시그널 연결들 아래쪽에 추가
        self.chkErase.toggled.connect(lambda v: self.view.set_erase_enabled(bool(v)))
        self.view.selectionMade.connect(self.on_selection_erase)

        # 톤 컨트롤 변경 시 즉시 재처리
        def _on_tone_control_changed():
            self._reprocess_last()

        self.chkHistNorm.toggled.connect(_on_tone_control_changed)
        self.sldNormStrength.valueChanged.connect(_on_tone_control_changed)

        # 파일 로드용 PixelFormat 콤보 추가
        self._init_load_pixfmt_combo()
        fAcq.addRow(QLabel("PixelFormat (Load)"), self.cmbLoadPixfmt)

        # 슬라이더는 기본 비활성/숨김 (원하면 노출)
        self.sldNormStrength.setEnabled(True)
        self.sldNormStrength.hide()
        self.sldNormStrength.setEnabled(False)

        # 초기 렌더 갱신
        self._reprocess_last()

    def _erase_points_in_rect(self, pts: list[tuple[int, int]], rect_xywh: tuple[int, int, int, int]) -> list[
        tuple[int, int]]:
        """rect(이미지 좌표계) 내부의 (x,y) 포인트를 제거한 새 리스트 반환."""
        if not pts:
            return []
        x, y, w, h = rect_xywh
        if w <= 0 or h <= 0:
            return list(pts)
        # inclusive 경계
        l, t = x, y
        r, b = x + w - 1, y + h - 1
        out = []
        for (px, py) in pts:
            if not (l <= px <= r and t <= py <= b):
                out.append((px, py))
        return out

    def _rebuild_overlay_from_points_only(self):
        """clusters 무시하고, 현재 self._dark_pts + self._bright_pts 만 1×1로 표시."""
        pts_all = (self._dark_pts or []) + (self._bright_pts or [])
        self._apply_clusters_to_overlay(ClusterDataList([]), points=pts_all)
        self.refresh_view()

    def _sync_latest_after_erase(self):
        """_latest_dark/_latest_bright 의 pts 를 현재 포인트로 동기화 (CSV 반영)."""
        if isinstance(self._latest_dark, dict):
            self._latest_dark["pts"] = list(self._dark_pts or [])
        if isinstance(self._latest_bright, dict):
            self._latest_bright["pts"] = list(self._bright_pts or [])

    def on_selection_erase(self, x: int, y: int, w: int, h: int):
        """GridView.selectionMade → 해당 영역의 디펙 포인트 제거 후 UI/CSV 반영."""
        from PyQt5.QtWidgets import QMessageBox

        # 표시용 포인트가 비어있고, 보관본만 있다면 보관본에서 끌어옴
        if not self._dark_pts and self._latest_dark.get("pts"):
            self._dark_pts = list(self._latest_dark["pts"])
        if not self._bright_pts and self._latest_bright.get("pts"):
            self._bright_pts = list(self._latest_bright["pts"])

        if not (self._dark_pts or self._bright_pts):
            QMessageBox.information(self, "Erase", "지울 디펙이 없습니다. 먼저 Search를 실행하세요.")
            return

        before_dark = len(self._dark_pts)
        before_bright = len(self._bright_pts)

        self._dark_pts = self._erase_points_in_rect(self._dark_pts, (x, y, w, h))
        self._bright_pts = self._erase_points_in_rect(self._bright_pts, (x, y, w, h))

        removed_dark = before_dark - len(self._dark_pts)
        removed_bright = before_bright - len(self._bright_pts)

        # 클러스터 무효화 후 포인트만으로 오버레이 재구성
        self._dark_clusters = None
        self._bright_clusters = None
        self.lblDarkCnt.setText(f"Dark: {len(self._dark_pts)}")
        self.lblBrightCnt.setText(f"Bright: {len(self._bright_pts)}")

        # CSV 보관본 동기화
        self._sync_latest_after_erase()

        # 오버레이 갱신
        self._rebuild_overlay_from_points_only()

        QMessageBox.information(self, "Erase",
                                f"Removed {removed_dark} dark, {removed_bright} bright pixels in selection.")

    def _rebuild_overlay_from_state(self) -> None:
        """현재 저장된 디펙 결과(_dark/_bright)로 오버레이를 다시 구성."""
        any_done = False
        if self._dark_clusters is not None or self._dark_pts:
            self._apply_clusters_to_overlay(self._dark_clusters or ClusterDataList([]),
                                            points=self._dark_pts or [])
            any_done = True
        if self._bright_clusters is not None or self._bright_pts:
            self._apply_clusters_to_overlay(self._bright_clusters or ClusterDataList([]),
                                            points=self._bright_pts or [])
            any_done = True
        if not any_done:
            # 없으면 비우기
            self._overlay_rects = []
            if hasattr(self._overlay, "clear"):
                self._overlay.clear()
            self.refresh_view()

    def _init_tone_controls(self) -> None:
        """
        톤 컨트롤 초기화:
        - Normalize ON: 슬라이더 비활성화, Normalize 적용
        - Normalize OFF: 슬라이더 활성화, Stretch 적용 (100%면 무보정)
        """
        self.chkHistNorm.setEnabled(True)
        self.sldNormStrength.setEnabled(False)  # 초기에는 비활성 (Normalize OFF이므로)
        self.sldNormStrength.setTracking(True)

        # 초기 상태: Normalize OFF / 슬라이더 100%
        self.chkHistNorm.setChecked(False)
        self.sldNormStrength.setValue(100)

        def _on_norm_toggled(checked: bool):
            # Normalize ON: 슬라이더 비활성
            # Normalize OFF: 슬라이더 활성 + 100%로 리셋
            self.sldNormStrength.setEnabled(not checked)
            if not checked:
                if self.sldNormStrength.value() != 100:
                    self.sldNormStrength.blockSignals(True)
                    self.sldNormStrength.setValue(100)
                    self.sldNormStrength.blockSignals(False)
            self._reprocess_last()

        def _on_slider_changed(_v: int):
            # 슬라이더 움직일 때 재처리만 수행
            self._reprocess_last()

        self.chkHistNorm.toggled.connect(_on_norm_toggled)
        self.sldNormStrength.valueChanged.connect(_on_slider_changed)

    def _reprocess_last(self) -> None:
        base = self._src_frame if self._src_frame is not None else self._last_frame
        if base is not None:
            self.on_frame(base.copy())

    def _hist_percent_to_range(self, img_u8_or_u16: np.ndarray, bits: int, percent: int) -> tuple[int, int]:
        """
        C: CHistogram::GetRangeCheck()의 Percent 분기와 동일.
        nStPixelNum = (nCountsum * (100 - percent)) / 200
        nEdPixelNum = nCountsum - nStPixelNum
        누적합으로 start/end DN 산정.
        """
        if img_u8_or_u16 is None or img_u8_or_u16.size == 0:
            return 0, (1 << bits) - 1
        nMax = (1 << bits) - 1
        flat = img_u8_or_u16.reshape(-1)
        hist = np.bincount(flat, minlength=nMax + 1).astype(np.uint32)
        total = int(hist.sum())
        if total <= 0:
            return 0, nMax

        nStPixelNum = (total * (100 - int(percent))) // 200
        nEdPixelNum = total - nStPixelNum

        # 누적합으로 start/end 찾기 (C와 동일 의미)
        cdf = np.cumsum(hist, dtype=np.uint64)
        # start: cdf > nStPixelNum 되는 최초 bin
        start = int(np.searchsorted(cdf, nStPixelNum + 1, side="left"))
        # end  : cdf >= nEdPixelNum 되는 최초 bin
        end = int(np.searchsorted(cdf, nEdPixelNum, side="left"))
        start = max(0, min(nMax, start))
        end = max(start + 1, min(nMax, end))
        return start, end

    def _stretch_c_identical(self, img: np.ndarray, lo: int, hi: int, bits: int) -> np.ndarray:
        """
        C: CHistogram::Stretch()와 동일.
        p<=lo → 0, p>=hi → nMax, 그 외 → (p-lo)*(nMax/(hi-lo))
        u8/u16, 모노/컬러 모두 지원(컬러는 채널별 독립 적용).
        """
        nMax = (1 << bits) - 1
        if lo < 0: lo = 0
        if hi <= lo: hi = lo + 1

        if img.ndim == 2:
            src = img.astype(np.float32, copy=False)
            out = np.empty_like(img)
            out[src <= lo] = 0
            out[src >= hi] = nMax
            mid_mask = (src > lo) & (src < hi)
            scale = float(nMax) / float(hi - lo)
            out[mid_mask] = np.clip((src[mid_mask] - lo) * scale, 0, nMax).astype(img.dtype)
            return out

        # 컬러: 채널별 동일 적용
        out = np.empty_like(img)
        scale = float(nMax) / float(hi - lo)
        for c in range(img.shape[2]):
            src = img[..., c].astype(np.float32, copy=False)
            ch = np.empty_like(img[..., c])
            ch[src <= lo] = 0
            ch[src >= hi] = nMax
            mid_mask = (src > lo) & (src < hi)
            ch[mid_mask] = np.clip((src[mid_mask] - lo) * scale, 0, nMax).astype(img.dtype)
            out[..., c] = ch
        return out



    def _effective_pixfmt(self) -> str:
        """현재 프레임에 적용해야 하는 PixelFormat을 일관되게 반환."""
        try:
            pixfmt_cam = str(self.cam.get("PixelFormat") or "")
        except Exception:
            pixfmt_cam = ""
        if self._frame_source == "file" and self._assumed_load_pixfmt:
            return self._assumed_load_pixfmt
        return pixfmt_cam

    def _init_load_pixfmt_combo(self) -> None:
        """Open Image용 수동 PixelFormat 선택 콤보박스 생성."""
        self.cmbLoadPixfmt = QComboBox()
        self.cmbLoadPixfmt.addItems([
            "Auto (detect)",
            "Mono8", "Mono10", "Mono12", "Mono14", "Mono16",
            "BayerRG8", "BayerRG10", "BayerRG12", "BayerRG16",
            "BayerGR8", "BayerGR10", "BayerGR12", "BayerGR16",
            "BayerGB8", "BayerGB10", "BayerGB12", "BayerGB16",
            "BayerBG8", "BayerBG10", "BayerBG12", "BayerBG16",
            "RGB8", "RGB10", "RGB12", "RGB16",
        ])
        self.cmbLoadPixfmt.setCurrentText("Auto (detect)")
        self.cmbLoadPixfmt.setToolTip(
            "파일을 열 때 사용할 데이터 포맷.\n"
            "아는 경우 정확히 선택, 모르면 Auto(detect)."
        )

    # ───────────────────────── Acquisition ─────────────────────────
    def on_connect(self):
        with contextlib.suppress(Exception):
            self.cam.stop_preview()  # 혹시 켜져 있으면 끄기
        if not self.cam.connect():
            QMessageBox.critical(self, "Camera", "Camera not found/online.")
            return
        # PixelFormat 기반으로 Dark 범위/기본값 세팅
        try:
            pixfmt = str(self.cam.get("PixelFormat"))
        except Exception:
            pixfmt = None
        set_dark_spin_range_by_pixfmt(self.spnDark, pixfmt)
        QMessageBox.information(self, "Camera", "Connected.\nClick 'Snap' to grab a frame.")

    def _apply_histogram_normalize(self, bgr8: np.ndarray) -> np.ndarray:
        """
        C CHistogram::Normalize 동일 동작 (표시 전용):
          - 현재 프리뷰는 BGR8이므로 bits=8, nStartVal=0(Cut 미사용)로 적용
          - 결과를 그대로 반환 (블렌딩 없음)
        """
        if bgr8 is None or bgr8.size == 0:
            return bgr8
        # 프리뷰는 BGR8이므로 8비트 기준으로 LUT 적용
        return histogram_normalize_c_identical(bgr8, bits=8, nStartVal=0)

    def setup_open_image_button(self) -> None:
        """
        Acquisition 그룹(FormLayout)에 'Open Image' 버튼을 추가하고 on_open_image에 연결한다.
        • 기존 코드 변경을 최소화하기 위해 제목이 'Acquisition'인 QGroupBox를 찾아 사용.
        """
        # 버튼 생성
        self.btnOpen = QPushButton("Open Image")
        self.btnOpen.setToolTip("파일에서 이미지를 불러옵니다. (카메라 없이도 검사 가능)")
        self.btnOpen.clicked.connect(self.on_open_image)

        # 'Acquisition' 그룹 찾기
        acq_boxes = [g for g in self.findChildren(QGroupBox) if g.title() == "Acquisition"]
        if not acq_boxes:
            # 안전망: 패널 레이아웃 최상단에 별도 박스로 추가
            gb = QGroupBox("Acquisition")
            fl = QFormLayout();
            fl.addRow(self.btnOpen);
            gb.setLayout(fl)
            # 좌측 패널(첫 번째 QSplitter child)의 첫 번째 위젯을 찾아 붙임
            # 레이아웃 구조가 다를 수 있으므로 실패 시 버튼만 표시
            try:
                splitter = self.findChild(QSplitter)
                if splitter:
                    left_panel = splitter.widget(0)
                    if left_panel and isinstance(left_panel.layout(), QVBoxLayout):
                        left_panel.layout().insertWidget(1, gb)
            except Exception:
                self.btnOpen.setParent(self)  # 최후 수단
            return

        acq = acq_boxes[0]
        lay = acq.layout()
        if isinstance(lay, QFormLayout):
            lay.addRow(self.btnOpen)
        else:
            # 다른 레이아웃이면 맨 아래에 추가
            lay.addWidget(self.btnOpen)

    def _try_infer_pixfmt_from_file(self, path: str) -> Optional[str]:
        """
        TIFF/DNG 메타 기반 러프 추정 (정확도보다 '대략' 파악용).
        실패하면 None 리턴.
        """
        try:
            import tifffile as tiff
            with tiff.TiffFile(path) as tf:
                page = tf.pages[0]
                bits = int(page.tags.get('BitsPerSample', 16).value) if 'BitsPerSample' in page.tags else 16
                photo = getattr(page, 'photometric', None)
                pstr = (photo.name if photo else "").lower()
                if "rgb" in pstr or "ycbcr" in pstr:
                    return f"RGB{bits}"
                if "minisblack" in pstr or "miniswhite" in pstr:
                    return f"Mono{bits}"
                # DNG/RAW-like
                if 'CFAPattern' in page.tags or 'CFARepeatPatternDim' in page.tags:
                    # 정확히 파싱하려면 추가 구현, 우선 RGGB 가정
                    return f"BayerRG{bits}"
        except Exception:
            pass
        return None

    def _parse_bayer_pattern_from_pixfmt(self, pixfmt: str) -> str:
        """'BayerRG10' → 'RGGB' 등. 기본 RGGB."""
        s = (pixfmt or "").upper()
        if "BAYERRG" in s: return "RGGB"
        if "BAYERGR" in s: return "GRBG"
        if "BAYERGB" in s: return "GBRG"
        if "BAYERBG" in s: return "BGGR"
        return "RGGB"

    def _clear_defects_and_overlay(self, *, reset_counters: bool = True) -> None:
        """모든 디펙 상태와 오버레이를 완전히 초기화."""
        # 검출 포인트/클러스터 비움
        self._dark_pts.clear()
        self._bright_pts.clear()
        self._dark_clusters = None
        self._bright_clusters = None

        # 렌더러 내부 아이템 비움
        try:
            if self._overlay is not None and hasattr(self._overlay, "clear"):
                self._overlay.clear()
        except Exception:
            pass

        # UI 직그림용 사각형 목록 비움
        self._overlay_rects = []
        try:
            self.view.set_overlay_rects([])  # 즉시 비주얼도 비움
        except Exception:
            pass

        # 카운터 초기화
        if reset_counters:
            self.lblDarkCnt.setText("Dark: 0")
            self.lblBrightCnt.setText("Bright: 0")

        # 뷰 갱신
        self.refresh_view()

    def _clear_visuals_only(self) -> None:
        """보관된 최신 결과(_latest_*)는 유지하고, 화면 오버레이/카운터만 초기화."""
        # 화면용(현재 프레임 기준) 결과를 비움
        self._dark_pts.clear()
        self._bright_pts.clear()
        self._dark_clusters = None
        self._bright_clusters = None

        # 오버레이 렌더러 초기화
        try:
            if self._overlay is not None and hasattr(self._overlay, "clear"):
                self._overlay.clear()
        except Exception:
            pass

        self._overlay_rects = []
        try:
            self.view.set_overlay_rects([])
        except Exception:
            pass

        # 화면 카운터도 초기화 (보관 데이터와 무관)
        self.lblDarkCnt.setText("Dark: 0")
        self.lblBrightCnt.setText("Bright: 0")

        self.refresh_view()

    def on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.critical(self, "Open Image", "Failed to load the image.")
            return

        manual = ""
        if hasattr(self, "cmbLoadPixfmt") and self.cmbLoadPixfmt is not None:
            manual = (self.cmbLoadPixfmt.currentText() or "").strip()
            if manual.lower().startswith("auto"):
                manual = ""
        self._assumed_load_pixfmt = manual or self._try_infer_pixfmt_from_file(path) or ""
        if not self._assumed_load_pixfmt:
            QMessageBox.warning(
                self, "PixelFormat",
                "파일의 PixelFormat을 알 수 없습니다.\n"
                "왼쪽 패널 'PixelFormat (Load)'에서 수동으로 선택해 주세요."
            )
            return

        # ★ 새 이미지 들어오기 전에 '화면만' 초기화 (보관본은 유지)
        self._clear_visuals_only()

        self._frame_source = "file"
        self._src_frame = np.ascontiguousarray(img.copy())

        # 프레임 표시 업데이트
        self._reprocess_last()

        try:
            set_dark_spin_range_by_pixfmt(self.spnDark, self._assumed_load_pixfmt)
        except Exception:
            set_dark_spin_range_by_frame(self, img)

        QMessageBox.information(self, "Open Image", f"Loaded:\n{path}")

    def on_snap(self):
        ctrl = self.cam.ctrl
        try:
            if not ctrl.is_grabbing():
                ctrl.start_grab(buffer_count=4)

            frame = ctrl.get_next_frame(timeout_ms=2000)
            if frame is None:
                QMessageBox.warning(self, "Snap", "Timeout while grabbing a frame.")
                return

            # ★ 새 프레임 반영 전에 '화면만' 초기화 (보관본은 유지)
            self._clear_visuals_only()

            self._frame_source = "camera"
            self._src_frame = np.ascontiguousarray(frame.copy())
            self._reprocess_last()

        except Exception as e:
            QMessageBox.critical(self, "Snap", f"Failed: {e}")
        finally:
            with contextlib.suppress(Exception):
                ctrl.stop_grab(flush=False)

    # ───────────────────────── Frame / View ───────────────────────
    def _update_mean_brightness(self):
        """
        현재 프레임의 평균 밝기를 '네이티브 DN'으로 계산해 self.lblMean에 표시.
        """
        if self._last_frame is None:
            self.lblMean.setText("Mean (DN): -")
            return

        eff_pixfmt = self._effective_pixfmt()  # ★
        bits = pick_effective_bit_depth(eff_pixfmt, self._last_frame)

        fr = self._last_frame
        if fr.ndim == 2:
            if fr.dtype == np.uint16 and bits >= 10:
                gray_native = u16_to_native_dn(fr, bits)  # 0..max_native
            else:
                gray_native = fr if fr.dtype == np.uint8 else fr.astype(np.uint16)
        else:
            if fr.dtype == np.uint8:
                bgr = fr[..., :3].astype(np.float32)
                y = 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]
                gray_native = np.clip(y, 0, 255).astype(np.uint8)
            else:
                bgr = fr[..., :3].astype(np.float32)
                y = 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]
                gray_native = np.clip(y, 0, 65535).astype(np.uint16)
                if bits >= 10:
                    gray_native = u16_to_native_dn(gray_native, bits)

        mean_dn = float(gray_native.mean()) if gray_native.size else 0.0
        self.lblMean.setText(f"Mean: {mean_dn:.1f}")
        self._gray_native_for_hover = gray_native

    def on_hover(self, ix: int, iy: int):
        """GridView에서 보낸 호버 좌표를 받아 DN만 표시."""
        if ix < 0 or iy < 0 or self._last_bgr is None:
            self.lblHover.setText("x:- y:-  DN:-")
            return

        # 네이티브 DN (가능하면)
        dn_text = "-"
        gn = getattr(self, "_gray_native_for_hover", None)
        if gn is not None and iy < gn.shape[0] and ix < gn.shape[1]:
            dn_val = int(gn[iy, ix])
            eff_pixfmt = self._effective_pixfmt()  # ★
            bits = pick_effective_bit_depth(eff_pixfmt, self._last_frame)
            dn_text = f"{dn_val} (≈{bits}-bit)"

        self.lblHover.setText(f"x:{ix} y:{iy}  DN:{dn_text}")

    def _update_histogram_view(self, disp_bgr8: np.ndarray) -> None:
        """
        (5번) 히스토그램 갱신: 표시용 버퍼 기준으로 8bit 그레이 히스토그램 업데이트.
        히스토그램 위젯(self.hist)이 없으면 조용히 무시.
        """
        try:
            if disp_bgr8 is None or disp_bgr8.size == 0 or disp_bgr8.dtype != np.uint8:
                return
            if not hasattr(self, "hist") or self.hist is None:
                return
            g8 = cv2.cvtColor(disp_bgr8, cv2.COLOR_BGR2GRAY)
            # 표시 버퍼는 8bit이므로 bits=8로 고정
            self.hist.set_bits(8)
            self.hist.update_histogram(g8)
        except Exception:
            pass

    def _apply_tone(self, img: np.ndarray, bits: int, he_on: bool, percent: int) -> np.ndarray:
        from defect_detection import histogram_normalize_c_identical
        percent = max(1, min(100, int(percent)))

        if he_on:
            return histogram_normalize_c_identical(img.copy(), bits=bits, nStartVal=0)

        if percent >= 100:
            return img

        src = img.copy()
        lo, hi = self._hist_percent_to_range(src, bits, percent)
        return self._stretch_c_identical(src, lo, hi, bits)

    def on_frame(self, frame: np.ndarray):
        from defect_detection import bayer_to_flat, bayer_flatfield_equalize
        self._last_frame = frame  # 이미 copy된 입력

        eff_pixfmt = (self._effective_pixfmt() or "").lower()
        bits = pick_effective_bit_depth(eff_pixfmt, frame)
        he_on = bool(getattr(self, "chkHistNorm", None) and self.chkHistNorm.isChecked())
        percent = int(self.sldNormStrength.value()) if hasattr(self, "sldNormStrength") else 100

        # --- 단일채널 ---
        if frame.ndim == 2:
            if eff_pixfmt.startswith("bayer"):
                # ★ 베이어 RAW → (네이티브 정렬) → BayertoFlat(평탄화) → 톤 → 그레이 표시
                raw = frame
                if raw.dtype == np.uint16 and bits >= 10:
                    raw = u16_to_native_dn(raw, bits)

                # 1) 4-위상 평탄화 (디모자이킹 없음, 여전히 1채널)
                ffc = bayer_flatfield_equalize(raw)

                # 2) 톤 보정 (표시 전용)
                tone_bits = 8 if ffc.dtype == np.uint8 else bits
                ffc = self._apply_tone(ffc, bits=tone_bits, he_on=he_on, percent=percent)

                # 3) 그레이 → BGR로 래핑하여 디스플레이
                if ffc.dtype == np.uint8:
                    g8 = ffc
                else:
                    # 네이티브 DN → 표시용 8U
                    g8 = u16_to_u8_display(ffc.astype(np.uint16, copy=False), bits)
                disp = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

            else:
                # 기존 모노 처리 그대로
                if frame.dtype == np.uint16:
                    g16 = frame
                    if bits >= 10:
                        g16 = u16_to_native_dn(g16, bits)
                    g16 = self._apply_tone(g16, bits=bits, he_on=he_on, percent=percent)
                    g8 = u16_to_u8_display(g16, bits)
                    disp = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
                else:
                    g8 = self._apply_tone(frame, bits=8, he_on=he_on, percent=percent)
                    disp = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

        # --- 컬러 컨테이너 ---
        else:
            if frame.dtype == np.uint16:
                c16 = self._apply_tone(frame, bits=bits, he_on=he_on, percent=percent)
                scale = 255.0 / max(1, (1 << max(8, bits)) - 1)
                c8 = (c16.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
                disp = c8
            else:
                c8 = self._apply_tone(frame, bits=8, he_on=he_on, percent=percent)
                disp = c8

        self._last_bgr = disp
        self._update_histogram_view(disp)
        self._update_mean_brightness()
        self.refresh_view()

    def _composite_overlay_on_image(self, bgr_img: np.ndarray, rects: list) -> np.ndarray:
        """
        QPainter로 그리는 대신, 표시용 BGR 이미지 위에 직접 사각형/포인트를 그려 넣는다.
        rects: [(l, t, w, h, (B, G, R), size), ...]  // size: 1~5
        반환: 합성된 BGR 이미지 (원본은 보존)
        """
        if (
                bgr_img is None
                or bgr_img.size == 0
                or bgr_img.ndim != 3
                or bgr_img.shape[2] < 3
        ):
            return bgr_img

        h, w = bgr_img.shape[:2]
        out = bgr_img.copy()

        def _thickness(sz: int) -> int:
            # 너무 얇아 보이지 않게 1~6 범위로 매핑
            sz = int(max(1, min(5, sz)))
            return {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}[sz]

        # ── 사각형 클러스터 그리기 ──
        for item in (rects or []):
            if len(item) < 6:
                continue
            l, t, ww, hh, bgr, sz = item

            # 경계/정수 보정
            l = int(max(0, min(w - 1, l)))
            t = int(max(0, min(h - 1, t)))
            ww = int(max(1, ww))
            hh = int(max(1, hh))
            r = int(min(w - 1, l + ww - 1))
            b = int(min(h - 1, t + hh - 1))
            if r <= l or b <= t:
                continue

            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))  # 이미 BGR
            cv2.rectangle(
                out, (l, t), (r, b),
                color, thickness=_thickness(sz),
                lineType=cv2.LINE_AA
            )

        # ── 포인트(픽셀)도 확실히 보이도록 5×5 채움 사각형으로 찍기 ──
        dot_color = (0, 0, 255)  # BGR (red)
        # 두 리스트를 순차로 순회
        for pts in [(getattr(self, "_dark_pts", []) or []), (getattr(self, "_bright_pts", []) or [])]:
            for p in pts:
                if len(p) < 2:
                    continue
                x, y = int(p[0]), int(p[1])
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                x0, y0 = max(0, x - 2), max(0, y - 2)
                x1, y1 = min(w - 1, x + 2), min(h - 1, y + 2)
                cv2.rectangle(out, (x0, y0), (x1, y1), dot_color, thickness=-1, lineType=cv2.LINE_AA)

        return out

    def refresh_view(self):
        if self._last_bgr is None:
            return

        # on_frame()에서 이미 Normalize/Stretch가 적용된 표시용 버퍼를 사용
        disp = self._last_bgr

        if self.chkOverlay.isChecked():
            overlay_ok = (self._overlay is not None and hasattr(self._overlay, "render_on"))
            overlay = self._overlay if overlay_ok else None
            rects = getattr(self, "_overlay_rects", []) or []
        else:
            overlay = None
            rects = []

        # 순서 중요: 먼저 rects, 다음 프레임
        self.view.set_overlay_rects(rects)
        self.view.set_frame(disp, overlay)

        self.view.update()
        self.view.repaint()

    def _apply_clusters_to_overlay(self, clusters: ClusterDataList, points: list[tuple[int, int]] | None = None):
        """
        디펙 오버레이 구성 (픽셀 ONLY).
        • 각 클러스터의 '우하단 픽셀(r-1,b-1)'만 1×1로 표시
        • ClusterData.rect는 (l,t,r,b) exclusive → 실제 픽셀은 (r-1,b-1)
        • clusters가 비어 있고 points만 있으면 points를 1×1로 표시
        """
        # DrawFigure 내부는 쓰지 않으므로 항상 비웁니다.
        if hasattr(self._overlay, "clear"):
            self._overlay.clear()

        ui_rects: list[tuple[int, int, int, int, tuple[int, int, int], int]] = []

        # 1) 클러스터 → 우하단 픽셀 1×1
        if clusters and clusters.vecClusterData:
            for c in clusters.vecClusterData:
                bgr, _ = cluster_style(c)  # 색상만 사용
                l, t, r, b = c.rect
                px = max(0, int(r) - 1)
                py = max(0, int(b) - 1)
                ui_rects.append((px, py, 1, 1, (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1))

        # 2) 포인트만 있는 경우(백업 경로) → 1×1
        elif points:
            default_bgr = (0, 0, 255)
            for (x, y) in points:
                ui_rects.append((int(x), int(y), 1, 1, default_bgr, 1))

        # 반영
        self._overlay_rects = ui_rects
        self.view.set_overlay_rects(ui_rects)
        self.view.update()
        self.view.repaint()

    def on_dark_search(self):
        from defect_detection import (
            FindDarkFieldClusterRect_corr,
            rgb_flatfield_equalize,
            extract_green_channel,
            _to_gray_native,
        )
        from PyQt5.QtWidgets import QApplication, QMessageBox

        if self._src_frame is None and self._last_frame is None:
            self.on_snap()
            if self._src_frame is None and self._last_frame is None:
                return

        eff_pixfmt = (self._effective_pixfmt() or "").lower()
        src_bits = pick_effective_bit_depth(
            eff_pixfmt,
            self._src_frame if self._src_frame is not None else self._last_frame
        )
        is_rgb_like = eff_pixfmt.startswith(("rgb", "bgr"))

        try:
            n_range = int(self.cmbBlock.currentText())
            thr_dn = int(self.spnDark.value())
        except Exception:
            n_range, thr_dn = 32, 100

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            base = self._src_frame if self._src_frame is not None else self._last_frame
            img = base.copy()

            if img.ndim == 3 and img.shape[2] >= 3 and is_rgb_like:
                img = rgb_flatfield_equalize(img)
                img = extract_green_channel(img)

            if img.ndim == 2 and img.dtype == np.uint16 and src_bits >= 10:
                img = u16_to_native_dn(img, src_bits)

            if img.ndim == 3 and img.shape[2] >= 3:
                img, _ = _to_gray_native(img, src_bits=src_bits, force_first_channel=True)

            n, pts, clusters = FindDarkFieldClusterRect_corr(
                image=img,
                n_range=n_range,
                n_min_threshold=-thr_dn,
                n_max_threshold=thr_dn,
                src_bits=src_bits
            )

            # 화면 상태 업데이트
            self._dark_pts = pts.point
            self._dark_clusters = clusters
            self._apply_clusters_to_overlay(clusters, points=self._dark_pts)
            self.lblDarkCnt.setText(f"Dark: {n}")
            if not self.chkOverlay.isChecked():
                self.chkOverlay.setChecked(True)

            # ★ 보관본 갱신 (CSV용, 프레임 크기도 기록)
            h, w = img.shape[:2]
            self._latest_dark = {
                "pts": list(self._dark_pts),
                "clusters": clusters,
                "frame_size": (w, h),
            }

            QMessageBox.information(self, "Dark", f"Complete.\nDetected: {n}")

        except Exception as e:
            QMessageBox.critical(self, "Dark", f"Failed: {e}")
            self._dark_clusters = None
        finally:
            QApplication.restoreOverrideCursor()
            self.refresh_view()

    def on_bright_search(self):
        from defect_detection import (
            FindBrightFieldClusterRect_c_identical,
            bayer_flatfield_equalize,
            rgb_flatfield_equalize,
            extract_green_channel,
            _to_gray_native,
        )
        from PyQt5.QtWidgets import QApplication, QMessageBox

        if self._src_frame is None and self._last_frame is None:
            self.on_snap()
            if self._src_frame is None and self._last_frame is None:
                return

        eff_pixfmt = (self._effective_pixfmt() or "").lower()
        src_bits = pick_effective_bit_depth(
            eff_pixfmt,
            self._src_frame if self._src_frame is not None else self._last_frame
        )
        is_bayer = eff_pixfmt.startswith("bayer")
        is_rgb_like = eff_pixfmt.startswith(("rgb", "bgr"))

        try:
            n_range = int(self.cmbBlock.currentText())
            p = int(self.spnBrightPct.value())
        except Exception:
            n_range, p = 32, 15

        p_abs = abs(p)
        min_p, max_p = -p_abs, p_abs

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            base = self._src_frame if self._src_frame is not None else self._last_frame
            img = base.copy()

            if img.ndim == 3 and img.shape[2] >= 3 and is_rgb_like:
                img = rgb_flatfield_equalize(img)
                img = extract_green_channel(img)

            if img.ndim == 2 and img.dtype == np.uint16 and src_bits >= 10:
                img = u16_to_native_dn(img, src_bits)

            if img.ndim == 2 and is_bayer:
                img = bayer_flatfield_equalize(img)

            if img.ndim == 3 and img.shape[2] >= 3:
                img, _ = _to_gray_native(img, src_bits=src_bits, force_first_channel=True)

            n, pts, clusters = FindBrightFieldClusterRect_c_identical(
                image=img,
                n_range=n_range,
                n_min_percent=min_p,
                n_max_percent=max_p,
                src_bits=src_bits
            )

            # 화면 상태 업데이트
            self._bright_pts = pts.point
            self._bright_clusters = clusters
            self._apply_clusters_to_overlay(clusters, points=self._bright_pts)
            self.lblBrightCnt.setText(f"Bright: {n}")
            if not self.chkOverlay.isChecked():
                self.chkOverlay.setChecked(True)

            # ★ 보관본 갱신 (CSV용)
            h, w = img.shape[:2]
            self._latest_bright = {
                "pts": list(self._bright_pts),
                "clusters": clusters,
                "frame_size": (w, h),
            }

            QMessageBox.information(self, "Bright", f"Complete.\nDetected: {n}")

        except Exception as e:
            QMessageBox.critical(self, "Bright", f"Failed: {e}")
            self._bright_clusters = None
        finally:
            QApplication.restoreOverrideCursor()
            self.refresh_view()

    # ───────────────────────── CSV Export ────────────────────────
    def on_save_csv(self):
        """
        원본 C 스타일 CSV로 저장:
        :Vieworks Camera Defective Pixel Data
        :H,:V
        :Dark Image Size,W,H   (가능 시)
        :Bright Image Size,W,H (가능 시)
        :dark field defective pixel
        x,y
        ...
        :Bright field defective pixel
        x,y
        ...
        """
        dark_pts = self._latest_dark.get("pts") or []
        bright_pts = self._latest_bright.get("pts") or []

        if not dark_pts and not bright_pts:
            QMessageBox.information(self, "CSV", "Defect map is empty. Snap & Search first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "defects.csv", "CSV (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                f.write(":Vieworks Camera Defective Pixel Data\n")
                f.write(":H,:V\n")


                f.write(":dark field defective pixel\n")
                for (x, y) in dark_pts:
                    f.write(f"{x},{y}\n")

                f.write(":Bright field defective pixel\n")
                for (x, y) in bright_pts:
                    f.write(f"{x},{y}\n")

            QMessageBox.information(self, "CSV", "Saved")
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Failed to save: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DefectTool()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec_())

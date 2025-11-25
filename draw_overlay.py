# -*- coding: utf-8 -*-
"""
draw_overlay.py
---------------
Overlay renderer for defect visualization.

• DefectLoc: (RGB color, size grade 1~5, rect=(l,t,r,b) exclusive, optional label)
• DrawFigure:
    - insert / extend / clear / count / is_empty
    - render_on(image, alpha=0.6, draw_centers=False, antialias=True)
• 8-bit / 16-bit 입력 모두 지원:
    - 3채널(BGR) 또는 단일채널(Gray) 이미지
    - 16U 입력은 표시용으로 8U로 안전 다운스케일 후 그립니다.
• Rect는 (left, top, right, bottom)에서 right/bottom은 **exclusive**
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

import numpy as np
import cv2


# ────────────────────────────────────────────────────────────────────────
# Data structure
# ────────────────────────────────────────────────────────────────────────
@dataclass
class DefectLoc:
    """
    하나의 디펙(또는 클러스터) 표시 정보.

    Attributes
    ----------
    color : Tuple[int,int,int]
        RGB 색상 (예: (0,0,255) = Blue). OpenCV는 BGR이므로 내부에서 변환합니다.
    size : int
        굵기 등급(1~5). 1이 가장 얇고, 5가 가장 두껍습니다.
    rect : Tuple[int,int,int,int]
        (left, top, right, bottom) – right/bottom은 **exclusive** 기준.
    label : Optional[str]
        사각형 왼상단에 그릴 짧은 텍스트(선택).
    """
    color: Tuple[int, int, int]
    size: int
    rect: Tuple[int, int, int, int]
    label: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────
# Overlay painter
# ────────────────────────────────────────────────────────────────────────
class DrawFigure:
    """
    디펙 오버레이 리스트를 관리하고, 주어진 프레임 위에 반투명으로 합성하는 렌더러.

    사용 예
    -------
    fig = DrawFigure()
    fig.insert(DefectLoc((0,0,255), 2, (100,100,120,140)))
    out = fig.render_on(frame, alpha=0.55)
    """

    def __init__(self) -> None:
        self.items: List[DefectLoc] = []

    # ───────── collection ops ─────────
    def clear(self) -> None:
        self.items.clear()

    def insert(self, loc: DefectLoc) -> None:
        self.items.append(loc)

    def extend(self, locs: Iterable[DefectLoc]) -> None:
        self.items.extend(locs)

    def count(self) -> int:
        return len(self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    # ───────── drawing helpers ─────────
    @staticmethod
    def _to_u8_for_display(img: np.ndarray) -> np.ndarray:
        """
        표시용 8-bit로 안전 변환.
        - 8U: 그대로 복사
        - 16U: 8U로 다운스케일 (>>8)
        - Gray: BGR로 승격하여 컬러 오버레이 가능하게 함
        """
        if img.ndim == 2:  # Gray
            if img.dtype == np.uint16:
                g8 = (img >> 8).astype(np.uint8)
            elif img.dtype == np.uint8:
                g8 = img
            else:
                g8 = np.clip(img, 0, 255).astype(np.uint8)
            return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

        # 3채널(BGR)
        if img.dtype == np.uint16:
            return (img >> 8).astype(np.uint8)
        if img.dtype == np.uint8:
            return img.copy()
        # 기타 -> 안전 변환
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r, g, b = rgb
        return (b, g, r)

    @staticmethod
    def _thickness_from_size(size: int) -> int:
        """
        size(1~5) → OpenCV line thickness.
        얇은 축소뷰에서도 보이도록 1~6 범위로 약간 강조.
        """
        size = int(max(1, min(5, size)))
        return {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}[size]

    def _draw_rects(
        self,
        canvas: np.ndarray,
        antialias: bool = True,
        draw_centers: bool = False,
    ) -> None:
        """
        모든 DefectLoc 사각형을 canvas 위에 그립니다(제자리 수정).
        """
        h, w = canvas.shape[:2]
        line_type = cv2.LINE_AA if antialias else cv2.LINE_8

        for it in self.items:
            l, t, r, b = it.rect
            # clip
            if r <= 0 or b <= 0 or l >= w or t >= h:
                continue
            l = max(0, min(w - 1, l))
            t = max(0, min(h - 1, t))
            r = max(1, min(w, r))
            b = max(1, min(h, b))
            if r <= l or b <= t:
                continue

            color_bgr = self._rgb_to_bgr(it.color)
            thickness = self._thickness_from_size(it.size)

            cv2.rectangle(
                canvas,
                (l, t),
                (r - 1, b - 1),
                color_bgr,
                thickness,
                lineType=line_type
            )

            # 선택: 중심 표시(십자)
            if draw_centers:
                cx = (l + r) // 2
                cy = (t + b) // 2
                s = max(1, thickness)
                cv2.drawMarker(
                    canvas, (cx, cy),
                    color_bgr,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=6 + 2 * s,
                    thickness=max(1, thickness // 2),
                    line_type=line_type
                )

            # 선택: 라벨
            if it.label:
                # 라벨 배경(약한 테두리) + 텍스트
                (tw, th), baseline = cv2.getTextSize(
                    it.label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                pad = 4
                bg_l = l
                bg_t = max(0, t - th - baseline - 2 * pad)
                bg_r = min(w, l + tw + 2 * pad)
                bg_b = min(h, bg_t + th + baseline + 2 * pad)

                cv2.rectangle(canvas, (bg_l, bg_t), (bg_r, bg_b),
                              (0, 0, 0), thickness=-1, lineType=line_type)
                cv2.rectangle(canvas, (bg_l, bg_t), (bg_r, bg_b),
                              color_bgr, thickness=1, lineType=line_type)
                cv2.putText(canvas, it.label, (bg_l + pad, bg_b - pad - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, line_type)

    # ───────── public API ─────────
    def render_on(
        self,
        image_bgr_or_gray: np.ndarray,
        *,
        alpha: float = 0.6,
        draw_centers: bool = False,
        antialias: bool = True,
    ) -> np.ndarray:
        """
        입력 이미지 위에 오버레이를 합성하여 반환합니다.

        Parameters
        ----------
        image_bgr_or_gray : np.ndarray
            원본 프레임 (BGR 8/16U 또는 Gray 8/16U).
        alpha : float
            0..1 범위의 오버레이 가중치. 0.6 정도가 보기 좋습니다.
        draw_centers : bool
            각 사각형 중심에 십자 마커를 그립니다.
        antialias : bool
            AA 라인 사용 여부.

        Returns
        -------
        np.ndarray (BGR 8U)
            합성 결과(표시용 8U). 원본 배열은 수정하지 않습니다.
        """
        if image_bgr_or_gray is None:
            raise ValueError("image_bgr_or_gray is None")

        base8 = self._to_u8_for_display(image_bgr_or_gray)
        if self.is_empty():
            return base8  # 표시만

        overlay = base8.copy()
        self._draw_rects(overlay, antialias=antialias, draw_centers=draw_centers)

        # 알파 합성
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return base8
        if alpha >= 1.0:
            return overlay

        # cv2.addWeighted: dst = src1*alpha + src2*(1-alpha) + gamma
        out = cv2.addWeighted(overlay, alpha, base8, 1.0 - alpha, 0.0)
        return out

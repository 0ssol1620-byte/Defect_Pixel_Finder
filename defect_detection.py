# -*- coding: utf-8 -*-
"""
defect_detection.py
-------------------
• 원본 C(수정 모듈)과 동일한 의미를 갖도록 8bit/16bit 경로를 명시 분기합니다.
  - Dark: data > (avgTile + thresholdDN)  (8U/16U 각각 네이티브 비교)
  - Bright: 임계는 타일 평균의 선형보간(avg_lin) 기반 ±% , 스코어는 타일 평균(avg_nn) 기반
• 빠른 타일 평균: NumPy 적분영상(u64 누적 + i64 산술) → OpenCV 빌드 의존 제거
• BGR16→Mono16: 원본 C와 동일하게 ×16 배율 적용
• 베이어 RAW → BGR: Edge Aware(EA) 지원 빌드는 EA 사용, 아니면 기본 변환
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (C++ 구조체 대응)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClusterData:
    rect: Tuple[int, int, int, int]
    count: int
    diffsum: float

@dataclass
class ClusterDataList:
    vecClusterData: List[ClusterData] = field(default_factory=list)

@dataclass
class DefectPT:
    point: List[Tuple[int, int]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Bayer RAW → BGR (8/16U)
# ─────────────────────────────────────────────────────────────────────────────
def bayer_to_flat(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Bayer RAW(8/16U) → BGR(동일 비트심도) 디모자이크.
    Edge-Aware(EA)가 지원되면 EA 사용, 아니면 기본 변환으로 fallback.
    pattern: 'RGGB'/'GRBG'/'GBRG'/'BGGR'
    """
    if raw.ndim != 2:
        return raw  # 이미 컬러면 통과
    if raw.dtype not in (np.uint8, np.uint16):
        raise ValueError("RAW Bayer must be uint8 or uint16")

    p = pattern.upper()
    m_basic = {
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
        "BGGR": cv2.COLOR_BayerBG2BGR,
    }
    m_ea = {
        "RGGB": cv2.COLOR_BayerRG2BGR_EA,
        "GRBG": cv2.COLOR_BayerGR2BGR_EA,
        "GBRG": cv2.COLOR_BayerGB2BGR_EA,
        "BGGR": cv2.COLOR_BayerBG2BGR_EA,
    }
    code_basic = m_basic.get(p, cv2.COLOR_BayerRG2BGR)
    code_ea = m_ea.get(p, cv2.COLOR_BayerRG2BGR_EA)

    try:
        return cv2.cvtColor(raw, code_ea)
    except Exception:
        return cv2.cvtColor(raw, code_basic)


def _to_gray_native(img: np.ndarray, *, src_bits: int, force_first_channel: bool=False) -> Tuple[np.ndarray, str]:
    """
    원본 C와 동일하게 '네이티브 비트심도'로 그레이를 반환.
      • src_bits <= 8  → uint8
      • src_bits >= 10 → uint16
    3채널 입력이면:
      - force_first_channel=True  : 첫 채널(B 채널)만 사용 (C의 'Color 미체크'와 동일)
      - force_first_channel=False : 표준 그레이(Rec.709). RGB16→Mono16일 때 ×16 적용.
    반환: (gray, 'u8' or 'u16')
    """
    if img.ndim == 2:
        if src_bits <= 8:  return img.astype(np.uint8,  copy=False), 'u8'
        else:              return img.astype(np.uint16, copy=False), 'u16'

    if img.ndim == 3 and img.shape[2] >= 3:
        if force_first_channel:
            ch0 = img[..., 0]  # B 채널
            if src_bits <= 8:  return ch0.astype(np.uint8,  copy=False), 'u8'
            else:              return ch0.astype(np.uint16, copy=False), 'u16'

        # 표준 그레이(색 선언이 RGB*일 때)
        bgr = img[..., :3]
        if src_bits <= 8 or bgr.dtype == np.uint8:
            b, g, r = [bgr[..., i].astype(np.float32) for i in (0, 1, 2)]
            y8 = 0.0722*b + 0.7152*g + 0.2126*r
            return np.clip(y8, 0, 255).astype(np.uint8), 'u8'
        else:
            b, g, r = [bgr[..., i].astype(np.float32) for i in (0, 1, 2)]
            # C++ ConvertBGR16ToMono16 과 동일: 마지막에 ×16
            y16 = (0.0722*b + 0.7152*g + 0.2126*r) * 16.0
            return np.clip(y16, 0, 65535).astype(np.uint16), 'u16'

    raise ValueError("Unsupported image for _to_gray_native")
def _apply_border_ignore(mask: np.ndarray, border: int) -> np.ndarray:
    """프레임 외곽 border 픽셀을 검사에서 제외."""
    if border <= 0:
        return mask
    h, w = mask.shape
    mask[:border, :] = 0
    mask[h-border:, :] = 0
    mask[:, :border] = 0
    mask[:, w-border:] = 0
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# 타일 평균: NumPy 적분영상 (u64 누적 + i64 산술) → 빠르고 안전
# ─────────────────────────────────────────────────────────────────────────────
def _tile_mean_map(img: np.ndarray, tile: int) -> np.ndarray:
    """
    타일 평균 맵(float32)을 반환.
      - 입력 img: uint8 또는 uint16, 2D
      - tile: 블록 크기(>=1)
    """
    if tile <= 0:
        tile = 1
    if img.ndim != 2:
        raise ValueError("img must be 2D grayscale")

    h, w = img.shape
    ny = (h + tile - 1) // tile
    nx = (w + tile - 1) // tile

    # 적분영상: 누적합은 u64로, 사각형 합 계산은 i64로 (언더/오버플로 방지)
    integ = img.astype(np.uint64, copy=False).cumsum(axis=0).cumsum(axis=1)
    integ_pad = np.zeros((h + 1, w + 1), dtype=np.uint64)
    integ_pad[1:, 1:] = integ
    I = integ_pad.astype(np.int64, copy=False)

    out = np.empty((ny, nx), dtype=np.float32)
    for ty in range(ny):
        y0 = ty * tile
        y1 = min(y0 + tile, h)
        for tx in range(nx):
            x0 = tx * tile
            x1 = min(x0 + tile, w)
            s = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0])
            area = float((y1 - y0) * (x1 - x0))
            out[ty, tx] = (float(s) / area) if area > 0 else 0.0
    return out


def _upsample_bilinear(tile_map: np.ndarray, w: int, h: int) -> np.ndarray:
    """타일 평균 맵 → 전체 해상도로 **선형보간** (임계 계산용 avg_lin)."""
    return cv2.resize(tile_map, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _upsample_nearest(tile_map: np.ndarray, w: int, h: int) -> np.ndarray:
    """타일 평균 맵 → 전체 해상도로 **최근접 복제** (스코어용 avg_nn)."""
    return cv2.resize(tile_map, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _clusters_from_mask_and_score(bin_mask: np.ndarray, score: np.ndarray) -> ClusterDataList:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return ClusterDataList([])
    sums = np.bincount(labels.reshape(-1), weights=score.reshape(-1), minlength=n)
    out: List[ClusterData] = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        out.append(ClusterData(rect=(int(x), int(y), int(x+w), int(y+h)),
                               count=int(area), diffsum=float(sums[k])))
    return ClusterDataList(out)

def cluster_style(d: ClusterData) -> Tuple[Tuple[int, int, int], int]:
    """
    클러스터 심각도 색상은 '합계(diffsum)'가 아닌 '평균 스코어(= diffsum / count)' 기준.
    평균 스코어(%): <15 Blue, <25 Green, <30 Orange, else Red
    size: count <=1 →1, <=2 →2, <=4 →3, <=8 →4, else →5
    색은 BGR(OpenCV) 튜플.
    """
    avg = float(d.diffsum) / max(1, int(d.count))

    if avg < 15:
        color = (255, 0, 0)       # Blue (B, G, R)
    elif avg < 25:
        color = (0, 180, 0)       # Green
    elif avg < 30:
        color = (0, 80, 180)      # Orange-ish
    else:
        color = (0, 0, 255)       # Red

    if d.count <= 1:
        size = 1
    elif d.count <= 2:
        size = 2
    elif d.count <= 4:
        size = 3
    elif d.count <= 8:
        size = 4
    else:
        size = 5
    return color, size



def _upsample_bilinear_centered_var(tile_map: np.ndarray, w: int, h: int, tile: int) -> np.ndarray:
    """
    타일 평균(ny×nx, 각 샘플은 '타일 중심')을 원해상도(h×w)로
    '중심 정렬' 선형보간. 마지막 부분 타일까지 정확히 반영.
    """
    ny, nx = tile_map.shape[:2]
    tile = max(1, int(tile))

    # 실제 타일 경계(마지막 타일은 부분 타일일 수 있음)
    x_edges = np.concatenate([np.arange(0, w, tile, dtype=np.float32), np.array([w], dtype=np.float32)])
    y_edges = np.concatenate([np.arange(0, h, tile, dtype=np.float32), np.array([h], dtype=np.float32)])
    x_edges = x_edges[:nx + 1]; x_edges[-1] = float(w)
    y_edges = y_edges[:ny + 1]; y_edges[-1] = float(h)

    # 타일 중심 좌표(픽셀 좌표계)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # 픽셀 중심 → 타일 인덱스 좌표로 사상
    up_x = np.arange(nx, dtype=np.float32)
    up_y = np.arange(ny, dtype=np.float32)
    xc = (np.arange(w, dtype=np.float32) + 0.5)
    yc = (np.arange(h, dtype=np.float32) + 0.5)
    u = np.interp(xc, x_centers, up_x, left=0.0, right=float(nx - 1))
    v = np.interp(yc, y_centers, up_y, left=0.0, right=float(ny - 1))
    map_x, map_y = np.meshgrid(u, v)

    out = cv2.remap(tile_map.astype(np.float32),
                    map_x.astype(np.float32),
                    map_y.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32)
def _inc_power_of_two(n_range: int) -> int:
    import math
    if n_range <= 1:
        return 1
    ins = int(math.log(n_range, 2) + 0.5)  # round-to-nearest
    inc = int(round(2.0 ** ins))
    return max(1, inc)

def _avg_block_grid(gray: np.ndarray, inc: int) -> Tuple[np.ndarray, int, int]:
    """
    CDefectCorr::GetAvgBlock 과 동일 의미:
    - inc 간격으로 블록을 썰고, '남는 끝 블록'은 실제 incy/incx 크기로 평균을 구해
      (ny × nx) 그리드에 저장.
    - 반환: (avg_grid[ny, nx], ny, nx)
    """
    h, w = gray.shape
    I = gray.astype(np.uint64, copy=False).cumsum(0).cumsum(1)
    Ipad = np.zeros((h + 1, w + 1), dtype=np.uint64)
    Ipad[1:, 1:] = I
    I64 = Ipad.astype(np.int64, copy=False)

    def rect_sum(x0: int, y0: int, x1: int, y1: int) -> int:
        return int(I64[y1, x1] - I64[y0, x1] - I64[y1, x0] + I64[y0, x0])

    ny = (h + inc - 1) // inc
    nx = (w + inc - 1) // inc
    grid = np.zeros((ny, nx), dtype=np.float32)

    by = 0
    for gy in range(ny):
        incy = inc if (by + inc) <= h else (h - by)
        bx = 0
        for gx in range(nx):
            incx = inc if (bx + inc) <= w else (w - bx)
            s = rect_sum(bx, by, bx + incx, by + incy)
            area = incx * incy
            grid[gy, gx] = (s / float(area)) if area > 0 else 0.0
            bx += inc
        by += inc

    return grid, ny, nx

def _avg_block_corr(gray: np.ndarray, inc: int) -> np.ndarray:
    """
    CDefectCorr::GetAvgBlock 동작을 그대로 재현.
    - 끝타일은 incx/incy = 남은 크기만큼
    - 각 타일의 평균(정수 나눗셈)을 '타일 영역'에 그대로 복제
    - 적분영상 뺄셈은 int64에서 수행(overflow 경고 방지)
    """
    h, w = gray.shape
    I = gray.astype(np.uint64, copy=False).cumsum(0).cumsum(1)
    Ipad = np.zeros((h + 1, w + 1), dtype=np.uint64)
    Ipad[1:, 1:] = I

    # ★ 뺄셈은 부호정수에서
    I64 = Ipad.astype(np.int64, copy=False)

    def rect_sum(x0: int, y0: int, x1: int, y1: int) -> int:
        return int(I64[y1, x1] - I64[y0, x1] - I64[y1, x0] + I64[y0, x0])

    out = np.zeros((h, w), dtype=np.int32)
    for y in range(0, h, inc):
        incy = inc if (y + inc) <= h else (h - y)
        for x in range(0, w, inc):
            incx = inc if (x + inc) <= w else (w - x)
            s = rect_sum(x, y, x + incx, y + incy)
            avg = s // (incx * incy) if (incx > 0 and incy > 0) else 0
            out[y:y + incy, x:x + incx] = avg
    return out


def FindDarkFieldClusterRect_corr(
    image: np.ndarray,
    n_range: int,
    n_min_threshold: int,
    n_max_threshold: int,
    *,
    src_bits: int,
) -> Tuple[int, DefectPT, ClusterDataList]:
    """
    CDefectCorr::FindDarkFieldDefect 동일:
      - inc = GetBlockIncrement(nRange)  (가장 가까운 2^k)
      - 평균 = GetAvgBlock (끝타일 incx/incy 실제 남은 크기, 정수 평균)
      - 판단: pix > avg + nMaxThreshold  또는  pix < avg + nMinThreshold
    """
    # 단일채널 네이티브 DN 준비
    if image.ndim == 3 and image.shape[2] >= 3:
        gray, _ = _to_gray_native(image, src_bits=src_bits)
    else:
        gray = image
    gray = (gray.astype(np.uint8, copy=False) if src_bits <= 8
            else gray.astype(np.uint16, copy=False))

    h, w = gray.shape
    inc = _inc_power_of_two(int(n_range))
    avg_map = _avg_block_corr(gray, inc)  # C GetAvgBlock과 동일 의미

    val_i32 = gray.astype(np.int32)
    hi = int(n_max_threshold)
    lo = int(n_min_threshold)

    # C 조건식
    hot_mask = ((val_i32 > (avg_map + hi)) | (val_i32 < (avg_map + lo))).astype(np.uint8)

    # 점수(로그): |100 - val*100/avg| (avg_map=0 보호)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.maximum(1.0, avg_map.astype(np.float32))
        score = np.where(hot_mask > 0,
                         np.abs(100.0 - (val_i32.astype(np.float32) * 100.0) / denom),
                         0.0).astype(np.float32)

    ys, xs = np.nonzero(hot_mask)
    pts = DefectPT(point=list(zip(xs.tolist(), ys.tolist())))

    clusters = _clusters_from_mask_and_score(hot_mask, score)
    return int(hot_mask.sum()), pts, clusters


def FindDarkFieldClusterRect_native(
    image: np.ndarray,
    n_threshold_dn: int,
    block_size: int,
    *,
    src_bits: int,
) -> Tuple[int, DefectPT, ClusterDataList]:
    """
    C(DefectMapMakingDlg::FindDarkdefect)과 1:1:
      • Mono8 :  pix >= thresholdDN
      • ≥10bit:  pix >  (avg + thresholdDN)
        (avg는 고정 inc×inc창 정수 평균, 끝타일은 시작점 되밀어 inc×inc로 유지)
    """
    # 입력 정규화
    if image.ndim == 3 and image.shape[2] >= 3:
        gray, _ = _to_gray_native(image, src_bits=src_bits)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Dark inspect expects single-channel image.")

    gray = (gray.astype(np.uint8, copy=False) if src_bits <= 8
            else gray.astype(np.uint16, copy=False))

    h, w = gray.shape
    inc = max(1, int(block_size))
    thr = int(max(0, n_threshold_dn))

    # Mono8: 평균 미사용, ≥
    if src_bits <= 8:
        hot_mask = (gray.astype(np.int32) >= thr).astype(np.uint8)
        diff = gray.astype(np.int32) - thr
        denom = max(1, thr)
        score = np.where(hot_mask > 0,
                         (diff.astype(np.float32) * 100.0) / float(denom),
                         0.0).astype(np.float32)
        ys, xs = np.nonzero(hot_mask)
        pts = DefectPT(point=list(zip(xs.tolist(), ys.tolist())))
        clusters = _clusters_from_mask_and_score(hot_mask, score)
        return int(hot_mask.sum()), pts, clusters

    # ≥10bit: avg는 '고정 inc×inc', 끝타일 시작점(sx,sy) 되밀기
    I = gray.astype(np.uint64, copy=False).cumsum(axis=0).cumsum(axis=1)
    Ipad = np.zeros((h + 1, w + 1), dtype=np.uint64)
    Ipad[1:, 1:] = I

    # ★ int64에서 뺄셈 (overflow 경고 방지)
    I64 = Ipad.astype(np.int64, copy=False)

    def rect_sum(x0: int, y0: int, x1: int, y1: int) -> int:
        return int(I64[y1, x1] - I64[y0, x1] - I64[y1, x0] + I64[y0, x0])

    avg_map = np.zeros((h, w), dtype=np.int32)
    y = 0
    while y < h:
        incy = inc if (y + inc) <= h else (h - y)
        sy = y if (y + inc) <= h else max(0, h - inc)
        x = 0
        while x < w:
            incx = inc if (x + inc) <= w else (w - x)
            sx = x if (x + inc) <= w else max(0, w - inc)
            s = rect_sum(sx, sy, sx + inc, sy + inc)
            avg = s // (inc * inc) if inc > 0 else 0  # 정수 평균
            avg_map[y:y + incy, x:x + incx] = int(avg)
            x += inc
        y += inc

    val_i32 = gray.astype(np.int32)
    hot_mask = (val_i32 > (avg_map + thr)).astype(np.uint8)

    denom = float(max(1, thr))
    score = np.where(hot_mask > 0,
                     ((val_i32 - avg_map).astype(np.float32) * 100.0) / denom,
                     0.0).astype(np.float32)

    ys, xs = np.nonzero(hot_mask)
    pts = DefectPT(point=list(zip(xs.tolist(), ys.tolist())))
    clusters = _clusters_from_mask_and_score(hot_mask, score)
    return int(hot_mask.sum()), pts, clusters







def bayer_flatfield_equalize(raw: np.ndarray) -> np.ndarray:
    """
    베이어 RAW(8U/16U) 각 위상(00,01,10,11)의 평균으로 gain을 만든 뒤,
    해당 위상 픽셀에 gain을 곱해 평탄화한다. (C의 BayertoFlat과 동일 의미)
    - 디모자이킹(색 보간) 절대 수행하지 않음.
    - dtype 한계(255/65535)에서 saturate.
    """
    if raw.ndim != 2 or raw.dtype not in (np.uint8, np.uint16):
        return raw
    h, w = raw.shape

    # 2x2 phase 평균
    avgs = np.empty((2, 2), dtype=np.float64)
    for oy in (0, 1):
        for ox in (0, 1):
            blk = raw[oy:h:2, ox:w:2]
            avgs[oy, ox] = float(blk.mean()) if blk.size else 1.0

    max_avg = float(avgs.max()) if avgs.max() > 0 else 1.0
    gains = np.where(avgs > 0, max_avg / avgs, 1.0)

    out = raw.astype(np.float64, copy=True)
    for oy in (0, 1):
        for ox in (0, 1):
            out[oy:h:2, ox:w:2] *= gains[oy, ox]

    max_level = 255 if raw.dtype == np.uint8 else 65535
    return np.clip(out, 0, max_level).astype(raw.dtype)

from functools import lru_cache

@lru_cache(maxsize=128)
def _tile_terms_template(incx: int, incy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    타일 내부 보간 항들을 템플릿으로 캐시.
    반환:
      x_term[incy,incx], y_term[incy,incx],
      x_is_left[incy,incx] (bool), y_is_up[incy,incx] (bool)
    C 동작 그대로: (xx - incx/2) < 0 인 쪽은 L-slope, 아니면 R-slope.
    반픽셀 보정: (0.5 if (inc%2) else 0.0)
    """
    incx = int(max(1, incx)); incy = int(max(1, incy))
    xx = np.arange(incx, dtype=np.float32)
    yy = np.arange(incy, dtype=np.float32)

    x_term = (xx - (incx / 2.0) + (0.5 if (incx % 2) else 0.0)).astype(np.float32)
    y_term = (yy - (incy / 2.0) + (0.5 if (incy % 2) else 0.0)).astype(np.float32)

    x_term = np.broadcast_to(x_term.reshape(1, incx), (incy, incx)).copy()
    y_term = np.broadcast_to(y_term.reshape(incy, 1), (incy, incx)).copy()

    x_is_left = (np.arange(incx, dtype=np.float32) - (incx / 2.0) < 0.0)
    y_is_up   = (np.arange(incy, dtype=np.float32) - (incy / 2.0) < 0.0)
    x_is_left = np.broadcast_to(x_is_left.reshape(1, incx), (incy, incx))
    y_is_up   = np.broadcast_to(y_is_up.reshape(incy, 1), (incy, incx))

    return (x_term, y_term, x_is_left.astype(np.bool_), y_is_up.astype(np.bool_))


def FindBrightFieldClusterRect_c_identical(
    image: np.ndarray,
    n_range: int,
    n_min_percent: int,
    n_max_percent: int,
    *,
    src_bits: int,
) -> Tuple[int, DefectPT, ClusterDataList]:
    """
    (고속 벡터화판) CDefectCorr::FindBrightFieldDefect 동등 의미.
    - 타일 그리드는 그대로, 타일 내부 보간/임계/판정은 벡터화
    - 임계: High = dAvg*(100+nMax)/100, Low = dAvg*(100+nMin)/100  (nMin이 음수면 ±p%)
    - 스코어: |100 - val*100 / (블록평균)|  ← 원본 C와 동일
    """
    # 1) 단일채널 네이티브 DN 준비
    if image.ndim == 3 and image.shape[2] >= 3:
        gray, _ = _to_gray_native(image, src_bits=src_bits, force_first_channel=True)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expect single-channel image (flatfielded if Bayer Raw).")

    gray = (gray.astype(np.uint8, copy=False) if src_bits <= 8
            else gray.astype(np.uint16, copy=False))

    h, w = gray.shape
    val32 = gray.astype(np.float32, copy=False)

    # 2) 블록 평균 그리드
    inc = _inc_power_of_two(int(n_range))
    avg_grid, ny, nx = _avg_block_grid(gray, inc)

    # 출력 버퍼
    dead = np.zeros((h, w), dtype=np.uint8)
    score = np.zeros((h, w), dtype=np.float32)

    # 공통 상수
    nmin = float(n_min_percent)
    nmax = float(n_max_percent)

    # 타일 루프(외부 루프만 유지, 내부는 벡터화)
    for gy in range(ny):
        y0 = gy * inc
        incy = inc if (y0 + inc) <= h else (h - y0)

        # y-slope 준비
        if gy == 0:
            avg_c_up   = avg_grid[min(gy+1, ny-1), :]
            avg_c_down = avg_grid[min(gy+1, ny-1), :]
            den_y = float(max(1, (incy // 2 + inc // 2)))
            yslopeL_gx = (avg_c_up  - avg_grid[gy, :]) / den_y
            yslopeR_gx = (avg_grid[gy, :] - avg_grid[gy, :]) / den_y  # 같은 값 대입되지만 형태 유지
            # 위 줄은 gy==0일 때 좌/우가 같은 값이라는 C 처리에 맞춰 아래에서 선택 시 동일해짐
            # 실제 값은 아래에서 gy==0 분기로 덮어씀
        elif gy == ny - 1:
            avg_c_up   = avg_grid[gy, :]
            avg_c_down = avg_grid[gy-1, :]
            den_y = float(max(1, (incy // 2 + inc // 2)))
            yslopeL_gx = (avg_grid[gy, :] - avg_c_down) / den_y
            yslopeR_gx = (avg_grid[gy, :] - avg_c_down) / den_y
        else:
            den_y = float(max(1, (incy // 2 + inc // 2)))
            yslopeL_gx = (avg_grid[gy, :] - avg_grid[gy-1, :]) / den_y
            yslopeR_gx = (avg_grid[gy+1, :] - avg_grid[gy, :]) / den_y

        for gx in range(nx):
            x0 = gx * inc
            incx = inc if (x0 + inc) <= w else (w - x0)

            # 이 블록의 중심 평균
            avg_c = float(avg_grid[gy, gx])

            # x-slope
            if gx == 0:
                den_x = float(max(1, (incx // 2 + inc // 2)))
                xslopeL = (avg_grid[gy, min(gx+1, nx-1)] - avg_c) / den_x
                xslopeR = xslopeL
            elif gx == nx - 1:
                den_x = float(max(1, (incx // 2 + inc // 2)))
                xslopeL = (avg_c - avg_grid[gy, gx-1]) / den_x
                xslopeR = xslopeL
            else:
                den_x = float(max(1, (incx // 2 + inc // 2)))
                xslopeL = (avg_c - avg_grid[gy, gx-1]) / den_x
                xslopeR = (avg_grid[gy, gx+1] - avg_c) / den_x

            # y-slope (gx 선택 반영)
            if gy == 0:
                yslopeL = (avg_grid[min(gy+1, ny-1), gx] - avg_c) / den_y
                yslopeR = yslopeL
            elif gy == ny - 1:
                yslopeL = (avg_c - avg_grid[gy-1, gx]) / den_y
                yslopeR = yslopeL
            else:
                yslopeL = (avg_c - avg_grid[gy-1, gx]) / den_y
                yslopeR = (avg_grid[gy+1, gx] - avg_c) / den_y

            # 타일 템플릿(벡터화용) 가져오기
            x_term, y_term, x_is_left, y_is_up = _tile_terms_template(incx, incy)

            # 슬로프 맵 (타일 전체에 대해 좌/우, 상/하 선택)
            slopeX = np.where(x_is_left, xslopeL, xslopeR).astype(np.float32)
            slopeY = np.where(y_is_up,   yslopeL, yslopeR).astype(np.float32)

            # 기대값 dAvg (타일 전체)
            dAvg = (avg_c + x_term * slopeX + y_term * slopeY).astype(np.float32)

            # 임계
            Th_H = (dAvg * (100.0 + nmax)) / 100.0
            Th_L = (dAvg * (100.0 + nmin)) / 100.0

            # 실제 값
            vb = val32[y0:y0+incy, x0:x0+incx]

            # 판정
            m = (vb > Th_H) | (vb < Th_L)

            # 기록
            if m.any():
                dead_block = dead[y0:y0+incy, x0:x0+incx]
                dead_block[m] = 1

                # 스코어는 블록 평균 기준 (C 동일)
                denom_block = max(1e-6, avg_c)
                sb = score[y0:y0+incy, x0:x0+incx]
                # 해당 픽셀만 계산
                sb[m] = np.abs(100.0 - (vb[m] * 100.0) / denom_block).astype(np.float32)

    # 포인트/클러스터 구성
    ys, xs = np.nonzero(dead)
    pts = DefectPT(point=list(zip(xs.tolist(), ys.tolist())))
    clusters = _clusters_from_mask_and_score(dead, score)
    return int(dead.sum()), pts, clusters


def rgb_flatfield_equalize(img: np.ndarray) -> np.ndarray:
    """
    RGB/BGR/RGBA 8U/16U 이미지의 채널 평균을 맞춰 게인 평탄화.
    - 입력: H×W×C (C>=3), dtype ∈ {uint8, uint16}
    - 동작: 각 채널 평균 -> max_avg/avg[c] 게인 -> 채널별 곱 -> 클립
    - 출력: 입력과 동일 dtype/shape
    """
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        return img
    if img.dtype not in (np.uint8, np.uint16):
        return img

    # 앞 3채널만 사용(BGR/RGB 모두 대응; 알파는 보존)
    base = img[..., :3].astype(np.float64)
    avgs = base.reshape(-1, 3).mean(axis=0)
    avgs = np.where(avgs > 0, avgs, 1.0)
    g = float(avgs.max())
    gains = g / avgs  # shape (3,)

    out = img.astype(np.float64, copy=True)
    for c in range(3):
        out[..., c] *= gains[c]

    max_level = 255 if img.dtype == np.uint8 else 65535
    out = np.clip(out, 0, max_level).astype(img.dtype)
    return out


def extract_green_channel(img: np.ndarray) -> np.ndarray:
    """
    인터리브 컬러(BGR/RGB/...)에서 G 채널만 추출해 단일채널로 반환.
    dtype/비트심도는 보존.
    """
    if img is None or img.ndim != 3 or img.shape[2] < 2:
        raise ValueError("extract_green_channel expects H×W×C image with C>=2.")
    return img[..., 1].copy()

def histogram_normalize_c_identical(img: np.ndarray, bits: int, nStartVal: int = 0) -> np.ndarray:
    """
    C CHistogram::Normalize 과 1:1 동작:
      value' = ((cdf[val] - nStartVal) / ((W*H) - nStartVal)) * nMax
    주의:
      - 분모는 항상 (W*H) (RGB여도 채널수 곱하지 않음)  ← C 코드 그대로
      - LUT 후 정수 캐스팅(양수 구간 → 내림) + [0, nMax] 클램프
      - 입력이 u8/u16 모두 지원, 컬러는 채널 공통 LUT 적용
    """
    if img is None or img.size == 0:
        return img

    nMax = (1 << int(bits)) - 1
    if nMax <= 0:
        return img

    # 원본 dtype 보관
    orig_dtype = img.dtype

    # 입력 값을 네이티브 DN 정수로 가정 (표시용 BGR8이면 8bit, 원본 u16이면 bits 기반)
    if img.ndim == 2:
        src = img
        H, W = src.shape
        total_hw = float(H * W)
        # 히스토그램 & CDF
        # (값 범위를 nMax로 고정; u8이어도 LUT 길이를 nMax+1로 맞춤)
        hist = np.bincount(src.reshape(-1), minlength=nMax + 1).astype(np.uint32)
        cdf = np.cumsum(hist, dtype=np.uint64)

        denom = max(1.0, total_hw - float(nStartVal))
        # LUT 구성 (double → 최종 T 캐스팅)
        lut = ((cdf.astype(np.float64) - float(nStartVal)) / denom) * float(nMax)
        lut = np.clip(lut, 0.0, float(nMax))

        # 적용
        out = lut[src].astype(orig_dtype, copy=False)
        return out

    elif img.ndim == 3 and img.shape[2] >= 3:
        H, W, C = img.shape
        total_hw = float(H * W)  # ★ 분모는 W*H (C와 동일)
        # 3채널 전체 샘플로 히스토그램 산출 (C의 GetHistogramEx(RGB)와 동치)
        flat = img.reshape(-1)
        hist = np.bincount(flat, minlength=nMax + 1).astype(np.uint32)
        cdf = np.cumsum(hist, dtype=np.uint64)

        denom = max(1.0, total_hw - float(nStartVal))
        lut = ((cdf.astype(np.float64) - float(nStartVal)) / denom) * float(nMax)
        lut = np.clip(lut, 0.0, float(nMax))

        bgr = [lut[img[..., k]].astype(orig_dtype, copy=False) for k in range(3)]
        out = np.stack(bgr, axis=-1)
        return out

    else:
        # 지원하지 않는 형태는 그대로 반환
        return img

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "ClusterData", "ClusterDataList", "DefectPT",
    "bayer_to_flat", "bayer_flatfield_equalize",
    "FindDarkFieldClusterRect_native",
    "FindDarkFieldClusterRect_corr",
    "FindBrightFieldClusterRect_c_identical",
    "histogram_normalize_c_identical",     # ← 추가
    "cluster_style",
]



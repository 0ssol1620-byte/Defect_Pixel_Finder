# core/camera_facade.py

from PyQt5.QtCore import QObject, pyqtSignal
from .camera_controller import CameraController
import numpy as np, time
from typing import Optional, Tuple


class CxpCamera(QObject):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, ctrl: Optional[CameraController] = None):
        super().__init__()
        self.ctrl = ctrl or CameraController()
        # [수정] CameraController에서 reshape된 프레임을 받기 위해 새 시그널에 연결
        self.ctrl.reshaped_frame_ready.connect(self.new_frame)

    # ---------- high-level API ----------
    def connect(self) -> bool:
        cams = self.ctrl.discover_cameras()
        if not cams:
            return False
        return self.ctrl.connect_camera_by_info(cams[0]["camera_info"])

    def set(self, name: str, value):
        self.ctrl.set_param(name, value)

    def get(self, name: str):
        return self.ctrl.get_param(name)

    def start_preview(self):
        self.ctrl.start_live_view()

    def stop_preview(self):
        self.ctrl.stop_live_view()

    def snap_pair(self, delay_ms: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """EMVA 전용: grab 두 장 연속 반환"""
        if not self.ctrl.is_grabbing():
            self.ctrl.start_grab(buffer_count=8)
            pre_started = True
        else:
            pre_started = False

        try:
            frame1 = self.ctrl.get_next_frame(timeout_ms=3000)
            if delay_ms:
                time.sleep(delay_ms / 1000)
            frame2 = self.ctrl.get_next_frame(timeout_ms=3000)
        finally:
            if pre_started:
                self.ctrl.stop_grab(flush=False)  # ← GRACEFUL 종료로 변경

        return frame1, frame2

    # convenience
    def width(self) -> int:
        return int(self.get("Width"))

    def height(self) -> int:
        return int(self.get("Height"))
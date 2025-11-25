# -*- coding: utf-8 -*-
"""
main.py — Defect Pixel Finder (PyQt5)

실행:
    python -m main
    또는
    python main.py --log-level INFO
"""

from __future__ import annotations
import sys
import os
import logging
import argparse

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# 앱 UI
from defect_tool_ui import DefectTool


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Defect Pixel Finder (PyQt5)")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="로깅 레벨 설정")
    p.add_argument("--highdpi", action="store_true",
                   help="고해상도 스케일링 강제 활성화")
    p.add_argument("--icon", default="vieworks.ico",
                   help="창/작업표시줄 아이콘 경로(.ico)")
    return p.parse_args(argv)


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def enable_high_dpi():
    try:
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def _set_windows_app_user_model_id(app_id: str) -> None:
    """
    Windows 작업표시줄에 아이콘이 정확히 반영되도록 AppUserModelID 설정.
    (다른 플랫폼에서는 무시)
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    if args.highdpi:
        enable_high_dpi()

    # ★ Windows 작업표시줄 아이콘 정확 적용
    _set_windows_app_user_model_id("Vieworks.DefectPixelFinder")

    app = QApplication(sys.argv)

    # ★ 앱/윈도우 아이콘 로드
    icon_path = args.icon
    if not os.path.isabs(icon_path):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), icon_path)
    app_icon = QIcon(icon_path)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)  # 작업표시줄/다수 다이얼로그 기본 아이콘

    # ★ 메인 윈도우
    w = DefectTool()
    w.setWindowTitle("Defect Pixel Finder")   # 제목표시줄 텍스트
    if not app_icon.isNull():
        w.setWindowIcon(app_icon)             # 제목표시줄(좌측) 아이콘

    w.resize(1200, 850)
    w.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

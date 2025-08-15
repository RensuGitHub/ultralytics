r"""
Simple webcam popup with live YOLO detections.

Requirements:
  - Python 3.8+
  - opencv-python
  - ultralytics (use local package or pip install ultralytics)

Usage (Windows cmd):
    - Using local repo (recommended inside this folder):
            python -m pip install -e .
            python -m pip install opencv-python
            python examples\webcam_popup.py --model yolov8n.pt --device cpu --conf 0.25

    - Or with pip package:
            python -m pip install ultralytics opencv-python
            python examples\webcam_popup.py --model yolov8n.pt

Notes:
  - Press 'q' or ESC to quit.
  - Use --camera to select a different camera index (default 0).
  - Use --imgsz to set inference size (default 640).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    print("Failed to import ultralytics. Install it or run `pip install -e .` in the repo root.")
    raise


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Webcam popup with YOLO detections")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights (.pt)")
    p.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' (default), 'auto' (prefers CUDA if available), or explicit CUDA ids like '0'.",
    )
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Class IDs to filter (optional)")
    p.add_argument("--show-fps", action="store_true", help="Show FPS on the window title")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Validate model path early for clearer errors
    mpath = Path(args.model)
    if not (mpath.exists() or mpath.with_suffix(".pt").exists() or mpath.with_suffix(".onnx").exists()):
        print(
            "Model not found: '",
            args.model,
            "'\n"
            "If you intended to use OpenVINO, export it first and pass the exported folder path:\n"
            "  yolo export model=yolov8n.pt format=openvino imgsz=640\n"
            "  python examples\\webcam_popup.py --model yolov8n_openvino_model\n"
            "Alternatively, use PyTorch weights (.pt):\n"
            "  python examples\\webcam_popup.py --model yolov8n.pt --device cpu\n",
            sep="",
        )
        return 1

    model = YOLO(args.model)

    # Resolve device with safe defaults (no NVIDIA GPU on this machine)
    try:
        import torch  # noqa: WPS433
        has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        has_cuda = False

    user_device = (args.device or "cpu").lower()
    if user_device == "auto":
        device = "cuda" if has_cuda else "cpu"
    elif user_device == "cuda":
        device = "cuda" if has_cuda else "cpu"
    else:
        device = user_device

    # Use Ultralytics streaming predictions and draw with result.plot()
    window_name = "YOLO Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # results is a generator when stream=True
    try:
        for result in model.predict(
            source=args.camera,
            stream=True,
            show=False,  # we'll draw manually to control window
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            classes=args.classes,
            verbose=False,
        ):
            # result.plot() returns a BGR numpy image with boxes, masks, labels drawn
            frame = result.plot()

            if args.show_fps and hasattr(result, "speed"):
                # result.speed has 'preprocess', 'inference', 'postprocess' (ms)
                sp = result.speed
                total_ms = (sp.get("preprocess", 0) + sp.get("inference", 0) + sp.get("postprocess", 0))
                fps_txt = f" | ~{1000.0/total_ms:.1f} FPS" if total_ms > 0 else ""
                cv2.setWindowTitle(window_name, f"YOLO Webcam{fps_txt}")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or 'q'
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

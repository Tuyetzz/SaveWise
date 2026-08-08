"""One-time ONNX export. PRD 6.6.

Run this on the DESKTOP and copy the .onnx files to the Pi. ONNX artifacts are
portable, and desktop export avoids the ARM-specific export failures described
in PRD 6.5 as well as a slow, memory-hungry export on the SD card.

half=False is deliberate: ONNX Runtime's CPU execution provider has no fp16
fast path, so fp16 export gains nothing and can cost. If you need more speed,
reach for a smaller imgsz or int8 quantization, not fp16.
"""

from __future__ import annotations

from ultralytics import YOLO

EXPORTS = [
    ("yolo26n.pt", 480),  # scan: runs every frame
    ("yolo26s.pt", 640),  # confirm: candidates only
]


def main() -> None:
    for weights, imgsz in EXPORTS:
        print(f"exporting {weights} at imgsz={imgsz} ...")
        path = YOLO(weights).export(
            format="onnx", imgsz=imgsz, half=False, simplify=True
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()

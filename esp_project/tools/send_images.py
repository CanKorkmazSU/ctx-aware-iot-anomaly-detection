#!/usr/bin/env python3
"""Send images over USB serial to the ESP NanoFL demo.

Protocol (host -> ESP):
  magic  : 4 bytes  b"NFL0"
  len    : u16 LE   payload size in bytes
  payload: 25 x float32 LE (5x5 grayscale, normalized 0..1)

ESP replies with ASCII lines like:
  OK score=0.123456

Usage:
  python3 tools/send_images.py --port /dev/ttyACM0 path/to/img1.png path/to/img2.jpg

Dependencies:
  pip install pyserial pillow
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import serial  # type: ignore
from PIL import Image  # type: ignore


MAGIC = b"NFL0"


def image_to_5x5_floats(path: Path) -> list[float]:
    img = Image.open(path).convert("L")  # grayscale 8-bit
    img = img.resize((5, 5), resample=Image.BILINEAR)
    px = list(img.getdata())  # 25 ints 0..255
    return [p / 255.0 for p in px]


def build_frame(vals: list[float]) -> bytes:
    if len(vals) != 25:
        raise ValueError(f"expected 25 floats, got {len(vals)}")
    payload = struct.pack("<25f", *vals)
    return MAGIC + struct.pack("<H", len(payload)) + payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyACM0 or /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (ignored by USB-CDC on many chips)")
    ap.add_argument("--delay", type=float, default=0.05, help="Seconds to wait between frames")
    ap.add_argument("--wait-reply", action="store_true", help="Read one line reply per frame")
    ap.add_argument("images", nargs="+", help="Image paths")
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        # give ESP time to boot/reset when port opens
        time.sleep(0.5)
        ser.reset_input_buffer()

        for p in paths:
            vals = image_to_5x5_floats(p)
            frame = build_frame(vals)
            ser.write(frame)
            ser.flush()
            print(f"sent {p.name} ({len(frame)} bytes)")

            if args.wait_reply:
                line = ser.readline().decode(errors="replace").strip()
                print(f"  <- {line}")

            time.sleep(args.delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

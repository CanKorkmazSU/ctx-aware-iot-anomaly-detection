#!/usr/bin/bash

VIDEO=$1
FPS=$2
OUTPUT_DIR=$3

if [ -z "$VIDEO" ] || [ -z "$FPS" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: ./extract_samples.sh <video_path> <fps> <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Extracting frames from $VIDEO to $OUTPUT_DIR at $FPS FPS..."
ffmpeg -i "$VIDEO" -vf "fps=$FPS" "$OUTPUT_DIR/frame_%04d.png" -hide_banner -loglevel error
echo "Done."

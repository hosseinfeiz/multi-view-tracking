# run.sh
```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="data/BOXING_DATASET"
ENTRY="python track/main.py"
for TYPE in old; do
  for FOLDER in "$BASE/$TYPE"/*; do
    echo "Running on $FOLDER (camera‑params=$TYPE)"
    echo "y" | $ENTRY \
      --dataset_path "$FOLDER" \
      --camera_params "$TYPE" \
      --headless
  done
done
```

# Weights

- [Yolo](https://drive.google.com/file/d/1w1Vj-eXaiwf9Qjz4vuLemehCfikvmDlL/view?usp=sharing)
- [XMem](https://drive.google.com/file/d/1A4Eb0ML-mvUxRSg17t2FfMzyuhR1m1uJ/view?usp=sharing)
- [SAM](https://drive.google.com/file/d/13yzWus1aCbqHIdVLz2pl3tCGR6vzOxOV/view?usp=sharing)

# Usage
For creating the tracking_config.json you need to choose the target people for tracking in all the views. You should initialize the bounding boxes for the person in the first frame using the visualizer tool that will pop up during the inference if tracking_config.json doesn't exist in the video folder 
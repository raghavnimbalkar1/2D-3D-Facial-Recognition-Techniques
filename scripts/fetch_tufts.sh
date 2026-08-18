#!/bin/bash
# Fetch Tufts Face Database subsets (direct download, no form).
# TD_3D  : SfM-reconstructed PLY meshes, one per participant (~574 MB total)
# TD_RGB_E : 5-expression 2D photos per participant (~1.9 GB total)
# Terms: non-commercial research only, no redistribution; cite the TPAMI paper.
set -euo pipefail
BASE=https://tdface.ece.tufts.edu/downloads
OUT=${1:-data/raw}
mkdir -p "$OUT"/TD_3D "$OUT"/TD_RGB_E

for i in 1 2 3 4; do
  echo "[$(date '+%F %T')] downloading TD_3D_Set$i.zip ..."
  curl -L --retry 3 -o "$OUT/TD_3D/TD_3D_Set$i.zip" "$BASE/TD_3D/TD_3D_Set$i.zip"
  echo "[$(date '+%F %T')] extracting TD_3D_Set$i.zip ..."
  unzip -o -q "$OUT/TD_3D/TD_3D_Set$i.zip" -d "$OUT/TD_3D"
  rm "$OUT/TD_3D/TD_3D_Set$i.zip"
done

for i in 1 2 3 4; do
  echo "[$(date '+%F %T')] downloading TD_RGB_E_Set$i.zip ..."
  curl -L --retry 3 -o "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip" "$BASE/TD_RGB_E/TD_RGB_E_Set$i.zip"
  echo "[$(date '+%F %T')] extracting TD_RGB_E_Set$i.zip ..."
  unzip -o -q "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip" -d "$OUT/TD_RGB_E"
  rm "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip"
done

echo "[$(date '+%F %T')] DONE:"
du -sh "$OUT"/TD_3D "$OUT"/TD_RGB_E
find "$OUT/TD_3D" -name "*.ply" | wc -l
find "$OUT/TD_RGB_E" -type f | wc -l
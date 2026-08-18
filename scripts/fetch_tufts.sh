#!/bin/bash
set -euo pipefail
BASE=https://tdface.ece.tufts.edu/downloads
OUT=${1:-data/raw}
mkdir -p "$OUT"/TD_3D "$OUT"/TD_RGB_E

echo "=== Downloading 3D sets ==="
for i in 1 2 3 4; do
  if [ -f "$OUT/TD_3D/TD_3D_Set$i.zip" ] || [ ! -f "$OUT/TD_3D/TD_3D_$(( (i-1)*25 + 1 )).ply" ]; then
    echo "[$(date '+%F %T')] fetching TD_3D_Set$i.zip ..."
    curl -C - -L --retry 5 -o "$OUT/TD_3D/TD_3D_Set$i.zip" "$BASE/TD_3D/TD_3D_Set$i.zip"
    echo "[$(date '+%F %T')] extracting TD_3D_Set$i.zip ..."
    unzip -o -q "$OUT/TD_3D/TD_3D_Set$i.zip" -d "$OUT/TD_3D"
    rm -f "$OUT/TD_3D/TD_3D_Set$i.zip"
  else
    echo "TD_3D Set $i already present."
  fi
done

echo "=== Downloading RGB expression sets ==="
for i in 1 2 3 4; do
  echo "[$(date '+%F %T')] fetching TD_RGB_E_Set$i.zip ..."
  curl -C - -L --retry 5 -o "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip" "$BASE/TD_RGB_E/TD_RGB_E_Set$i.zip"
  echo "[$(date '+%F %T')] extracting TD_RGB_E_Set$i.zip ..."
  unzip -o -q "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip" -d "$OUT/TD_RGB_E"
  rm -f "$OUT/TD_RGB_E/TD_RGB_E_Set$i.zip"
done

echo "=== Download Complete ==="
echo "PLY meshes: $(find "$OUT/TD_3D" -name "*.ply" | wc -l)"
echo "RGB photos: $(find "$OUT/TD_RGB_E" -type f | wc -l)"
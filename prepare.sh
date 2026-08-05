#!/usr/bin/env bash
# Set up everything main.py needs: the Python packages, the four trained detectors, the four
# PTB-XL example records used for the local explanations, and the precomputed beat arrays the
# global analysis runs on.
#
# The beat arrays are 12 GB unpacked and are therefore hosted separately rather than in this
# repository. The archive is ~2 GB to download and needs 12 GB of free disk space once unpacked.
set -euo pipefail

cd "$(dirname "$0")"

# Download location of the precomputed beat arrays, a single zip archive holding one directory
# per pathology. A Dropbox share link must end in "?dl=1" to serve the file itself rather than
# the preview page.
BEATS_URL="https://www.dropbox.com/scl/fi/9z4uuypkbcbtcz9q8p65x/globalxaiecg_beats.zip?rlkey=wyu4ky07928sn1tuiofthr14g&dl=1"

# The trained detectors live in the AIME2024 repository and are fetched from there rather than
# vendored here, so both projects stay in sync with a single source of truth.
MODELS_BASE_URL="https://raw.githubusercontent.com/nilsgumpfer/AIME2024/main/models"

echo "== Installing Python packages =="
pip3 install -r requirements.txt

echo
echo "== Downloading trained detectors =="
# One Keras model.json + weights.h5 per pathology, from the AIME2024 study repository.
mkdir -p models
for P in AVB ISCH RBBB LBBB; do
  mkdir -p "models/$P"
  for F in model.json weights.h5; do
    if [ -f "models/$P/$F" ]; then
      echo "  $P/$F already present, skipping"
    else
      curl -fsSL -o "models/$P/$F" "${MODELS_BASE_URL}/${P}/${F}"
      echo "  $P/$F"
    fi
  done
done

echo
echo "== Downloading PTB-XL example records =="
# The four records explained in the local-explanation figures, from PTB-XL v1.0.3.
mkdir -p examples/ecgs
for REC in 03000/03509_hr 12000/12131_hr 14000/14493_hr 02000/02906_hr; do
  for EXT in hea dat; do
    NAME="$(basename "$REC").$EXT"
    if [ -f "examples/ecgs/$NAME" ]; then
      echo "  $NAME already present, skipping"
    else
      curl -fsSL -o "examples/ecgs/$NAME" "https://physionet.org/files/ptb-xl/1.0.3/records500/${REC}.${EXT}"
      echo "  $NAME"
    fi
  done
done

echo
echo "== Downloading precomputed beat arrays =="
mkdir -p results

MISSING=""
for P in AVB ISCH RBBB LBBB; do
  [ -f "results/$P/ECG_beats.npy" ] || MISSING="$MISSING $P"
done

if [ -z "$MISSING" ]; then
  echo "  All beat arrays already present, skipping"
elif [[ "$BEATS_URL" == TODO_* ]]; then
  echo "  No download URL configured yet. Set BEATS_URL at the top of this script." >&2
else
  echo "  Missing:$MISSING"
  echo "  Fetching archive (~2 GB, unpacks to 12 GB) ..."
  # -C - resumes a partial download, so an interrupted transfer of this size can be retried.
  curl -fL -C - -o results/beats.zip "$BEATS_URL"
  echo "  Unpacking ..."
  unzip -q -o results/beats.zip -d results/
  rm results/beats.zip
fi

echo
echo "Done. Run ./start.sh to reproduce the results."

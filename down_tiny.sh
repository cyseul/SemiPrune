#!/usr/bin/env bash
set -euo pipefail

URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
OUT_DIR="../data"
ZIP_NAME="tiny-imagenet-200.zip"
TARGET_DIR="${OUT_DIR}/tiny-imagenet-200"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

if [ ! -f "${ZIP_NAME}" ]; then
    wget -O "${ZIP_NAME}" "${URL}"
else
    echo "[skip] ${ZIP_NAME} already exists"
fi

if [ ! -d "${TARGET_DIR}" ]; then
    unzip -q "${ZIP_NAME}"
else
    echo "[skip] ${TARGET_DIR} already exists"
fi

echo "Done."
echo "Zip: ${OUT_DIR}/${ZIP_NAME}"
echo "Extracted: ${TARGET_DIR}"
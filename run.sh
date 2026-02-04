#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by flags)
HF_USERNAME_DEFAULT="mkieffer"
HF_REPO_NAME_DEFAULT="MedQA-USMLE"
PRIVATE_DEFAULT="false"

HF_USERNAME="$HF_USERNAME_DEFAULT"
HF_REPO_NAME="$HF_REPO_NAME_DEFAULT"
PRIVATE="$PRIVATE_DEFAULT"

usage() {
  cat <<'EOF'
Usage: ./run.sh [--username <HF_USERNAME>] [--repo <HF_REPO_NAME>] [--private <true|false>]

Examples:
  ./run.sh
  ./run.sh --username mkieffer --repo MedQA-USMLE --private true
EOF
}

# simple long-flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --username)
      HF_USERNAME="${2:-}"; shift 2 ;;
    --repo|--reponame|--repo-name)
      HF_REPO_NAME="${2:-}"; shift 2 ;;
    --private)
      PRIVATE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

# Normalize PRIVATE to lowercase
PRIVATE="$(echo "$PRIVATE" | tr '[:upper:]' '[:lower:]')"
if [[ "$PRIVATE" != "true" && "$PRIVATE" != "false" ]]; then
  echo "Error: --private must be 'true' or 'false' (got '$PRIVATE')" >&2
  exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download and extract MedQA-USMLE if the processed file doesn't exist
DATA_FILE="data/us_qbank.jsonl"
TRAIN_FILE="data/train.jsonl"
TEST_FILE="data/test.jsonl"
DEV_FILE="data/dev.jsonl"
GDRIVE_FILE_ID="1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw"
ZIP_PATH="data/medqa.zip"
EXTRACT_DIR="data/medqa_extract"

if [ ! -f "$DATA_FILE" ] || [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ] || [ ! -f "$DEV_FILE" ]; then
  if [ -f "$ZIP_PATH" ]; then
    if unzip -tq "$ZIP_PATH" >/dev/null 2>&1; then
      echo "Using existing $ZIP_PATH..."
    else
      echo "Existing $ZIP_PATH is not a valid zip; re-downloading..."
      rm -f "$ZIP_PATH"
    fi
  elif [ -f "medqa.zip" ]; then
    echo "Found medqa.zip in repo root; moving to $ZIP_PATH..."
    mv "medqa.zip" "$ZIP_PATH"
    if ! unzip -tq "$ZIP_PATH" >/dev/null 2>&1; then
      echo "Moved file is not a valid zip; re-downloading..."
      rm -f "$ZIP_PATH"
    fi
  fi

  if [ ! -f "$ZIP_PATH" ]; then
    echo "Downloading MedQA-USMLE zip..."
    python3 - "$GDRIVE_FILE_ID" "$ZIP_PATH" <<'PY'
import sys
import re
import urllib.request
import http.cookiejar

file_id, dest = sys.argv[1], sys.argv[2]

def _stream_to_file(resp, path):
    with open(path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

resp = opener.open(base_url)
content_type = resp.headers.get("Content-Type", "")
data = resp.read()

def _extract_confirm_token(html_text):
    match = re.search(r"confirm=([0-9A-Za-z_]+)", html_text)
    if match:
        return match.group(1)
    for cookie in cj:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None

if "text/html" in content_type.lower():
    text = data.decode("utf-8", "ignore")
    token = _extract_confirm_token(text)
    if token:
        url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
        resp = opener.open(url)
        _stream_to_file(resp, dest)
    else:
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        resp = opener.open(url)
        _stream_to_file(resp, dest)
else:
    with open(dest, "wb") as f:
        f.write(data)

# sanity check for HTML download
try:
    with open(dest, "rb") as f:
        head = f.read(2048).lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise RuntimeError("Downloaded file looks like HTML, not a zip.")
except Exception as exc:
    raise SystemExit(f"Download failed: {exc}")
PY
  fi

  echo "Extracting MedQA-USMLE zip..."
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR"

  EXTRACTED_FILE="$(find "$EXTRACT_DIR" -type f -name 'US_qbank.jsonl' | head -n 1)"
  if [ -z "$EXTRACTED_FILE" ]; then
    echo "Error: US_qbank.jsonl not found after extraction." >&2
    exit 1
  fi

  US_DIR="$(dirname "$EXTRACTED_FILE")"
  for name in "US_qbank.jsonl" "train.jsonl" "test.jsonl" "dev.jsonl"; do
    src="${US_DIR}/${name}"
    if [ ! -f "$src" ]; then
      echo "Error: ${name} not found after extraction." >&2
      exit 1
    fi
  done

  if [ ! -f "$DATA_FILE" ]; then
    mv "${US_DIR}/US_qbank.jsonl" "$DATA_FILE"
  else
    echo "data/us_qbank.jsonl already exists, keeping existing file."
  fi
  if [ ! -f "$TRAIN_FILE" ]; then
    mv "${US_DIR}/train.jsonl" "$TRAIN_FILE"
  else
    echo "data/train.jsonl already exists, keeping existing file."
  fi
  if [ ! -f "$TEST_FILE" ]; then
    mv "${US_DIR}/test.jsonl" "$TEST_FILE"
  else
    echo "data/test.jsonl already exists, keeping existing file."
  fi
  if [ ! -f "$DEV_FILE" ]; then
    mv "${US_DIR}/dev.jsonl" "$DEV_FILE"
  else
    echo "data/dev.jsonl already exists, keeping existing file."
  fi
  rm -rf "$EXTRACT_DIR"
  rm -f "$ZIP_PATH"
else
  echo "data/*.jsonl already exists, skipping download."
fi

echo "Downloads complete."

# Run the uploader, forwarding the args
python3 upload_to_hf.py \
  --hf_username "$HF_USERNAME" \
  --hf_repo_name "$HF_REPO_NAME" \
  --private "$PRIVATE"

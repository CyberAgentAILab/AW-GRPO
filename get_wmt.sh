#!/usr/bin/env bash

set -euo pipefail

DEST=${1:-dataset/wmt.en-ja}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

declare -A URLS=(
  [2024]="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz  wmttest2024.en-ja.all.xml"
  [2023]="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz            wmttest2023.en-ja.all.xml"
  [2022]="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.2.tar.gz             wmttest2022.en-ja.all.xml"
  [2021]="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz             newstest2021.en-ja.all.xml"
)

mkdir -p "$DEST"
for Y in "${!URLS[@]}"; do
  read -r URL XML <<<"${URLS[$Y]}"
  echo "▶ ${Y}: download & extract …"
  curl -L -o "$TMP/${Y}.tar.gz" "$URL"
  tar -xzf "$TMP/${Y}.tar.gz" -C "$TMP"
  FOUND=$(find "$TMP" -name "$XML" | head -n1)
  if [[ -z "$FOUND" ]]; then
    echo "  ⚠ ${XML} not found; the upstream URL/tag may have changed"; continue
  fi
  mv "$FOUND" "$DEST/$XML"
  echo "  ✓ ${DEST}/${XML}"
done

echo "✅ All done. Files are in $DEST"
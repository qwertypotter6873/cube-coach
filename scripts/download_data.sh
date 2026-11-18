#!/usr/bin/env bash
# 촬영 자료(GitHub Release data-v1)를 받아 data/raw/, data/real/ 에 놓고 SHA-256 으로 확인한다.
#   scripts/download_data.sh                     모두 (약 1.66GB)
#   scripts/download_data.sh real_labels.zip     골라서 (파일 이름은 SHA256SUMS.txt 와 같게)
# 이미 있는 파일은 덮어쓰지 않는다. latest2.mov 는 앞 180초만 담은 판이다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="https://github.com/qwertypotter6873/cube-coach/releases/download/data-v1"
TMP="$ROOT/data/raw/.download"   # data/raw/ 는 저장소에서 제외된다
ALL=(sexy_move.mov cube_unsolved.mov latest.mov latest2.mov bg_patches.npy real_labels.zip)
FILES=("${@:-${ALL[@]}}")
mkdir -p "$TMP" "$ROOT/data/raw" "$ROOT/data/real"

curl -L --fail --silent --show-error -o "$TMP/SHA256SUMS.txt" "$BASE/SHA256SUMS.txt"
for f in "${FILES[@]}"; do
  if [ ! -f "$TMP/$f" ]; then
    echo "받는 중: $f"
    curl -L --fail --show-error --progress-bar -o "$TMP/$f.part" "$BASE/$f"
    mv "$TMP/$f.part" "$TMP/$f"
  fi
done

# 받은 것만 확인한다 (macOS 는 shasum, 리눅스는 sha256sum)
pattern="$(printf ' %s$|' "${FILES[@]}")"
check() { if command -v sha256sum >/dev/null; then sha256sum -c -; else shasum -a 256 -c -; fi; }
(cd "$TMP" && grep -E "${pattern%|}" SHA256SUMS.txt | check)

for f in "${FILES[@]}"; do
  case "$f" in
    *.mov) [ -e "$ROOT/data/raw/$f" ] || cp "$TMP/$f" "$ROOT/data/raw/$f" ;;
    bg_patches.npy) [ -e "$ROOT/data/real/$f" ] || cp "$TMP/$f" "$ROOT/data/real/$f" ;;
    real_labels.zip) unzip -n -q "$TMP/$f" -d "$ROOT/data/real/" ;;
  esac
done
echo "끝: data/raw/, data/real/ (받은 원본은 $TMP 에 남겨 둠, 지워도 된다)"

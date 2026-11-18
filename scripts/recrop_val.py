"""검증 세트를 시험셋처럼 다시 자른다: v2 모델이 큐브로 본 영역에 맞춰 큐브가 절반쯤 차게 (앱의 관심영역 따라가기와 같은 규칙).

검증 세트(data/real/label_frames_val)는 프레임 고르기 도구의 채도 기준 영역 그대로라 큐브가 자른 그림의 4분의 1쯤만 찼다
(정답 점 퍼짐 중앙값 0.24, 시험셋 0.54). 시험셋은 v2 모델로 다시 잘랐으므로(README 5.2절), 같은 방법으로 다시 잘라
'앱이 모델에 넣는 크기'에서 입력 크기를 견줄 수 있게 한다. 정답 점은 원본 프레임 좌표로 옮겨 새 자르기에 맞춘다
(라벨을 자르기에 쓰지 않는다. 새 영역 밖으로 나간 점은 eval_real_compare 가 '그 밖'으로 읽은 것으로 친다).
  새 관심영역 = cubecoach.facescan.follow_roi(v2 모델 192 입력의 분류 지도, 옛 영역) 를 세 번 되풀이
결과: data/real/label_frames_valc/ (720x720), data/real/labels_valc.json   → scripts/eval_real_compare.py --set valc
실행: .venv/bin/python scripts/recrop_val.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cubecoach.facescan import follow_roi
from cubecoach.render3d import N_CLASSES
from cubecoach.segnet import load_segnet

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data/real/label_frames_val", ROOT / "data/real/label_frames_valc"


def crop_frame(fr: np.ndarray, roi, out: int) -> np.ndarray:
    """roi(원본 너비 기준, 화면 중심 0)를 out x out 으로 자른다. 화면 밖은 검게 (label_real.py extract 와 같은 자르기)."""
    h, w = fr.shape[:2]
    side = roi[2] * w
    cx, cy = roi[0] * w + w / 2, roi[1] * w + h / 2
    k = out / side
    M = np.array([[k, 0, -(cx - side / 2) * k], [0, k, -(cy - side / 2) * k]], dtype=np.float32)
    return cv2.resize(cv2.warpAffine(fr, M, (out, out), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)), (out, out))


def main() -> None:
    m = load_segnet(ROOT / "models/cube_seg_net.pt")
    labels = json.loads((ROOT / "data/real/labels_val.json").read_text())
    frames = json.loads((SRC / "frames.json").read_text())
    DST.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / "data/raw" / frames[0]["video"]))
    out_meta, out_labels, spread = [], {}, []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, f["t"] * 1000)
        ok, fr = cap.read()
        assert ok, f["t"]
        h, w = fr.shape[:2]
        roi = tuple(f["roi"])
        for _ in range(3):  # 앱처럼 따라가며 수렴시킨다
            x = cv2.cvtColor(crop_frame(fr, roi, 192), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                o = m(torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255)[0]
            nr = follow_roi(o[:N_CLASSES].argmax(0).numpy(), roi)
            if nr is None:
                break
            roi = nr
        cv2.imwrite(str(DST / f["file"]), crop_frame(fr, roi, 720), [cv2.IMWRITE_JPEG_QUALITY, 92])
        out_meta.append({**f, "roi": [round(float(v), 5) for v in roi], "roi_before": f["roi"]})
        L = labels.get(f["file"])
        if L is None:
            continue
        o0 = f["roi"]
        pts = []
        for p in L["points"]:
            # 옛 자르기 좌표 → 원본 프레임 픽셀 → 새 자르기 좌표
            X = (o0[0] * w + w / 2) - o0[2] * w / 2 + p["x"] * o0[2] * w
            Y = (o0[1] * w + h / 2) - o0[2] * w / 2 + p["y"] * o0[2] * w
            nx = (X - ((roi[0] * w + w / 2) - roi[2] * w / 2)) / (roi[2] * w)
            ny = (Y - ((roi[1] * w + h / 2) - roi[2] * w / 2)) / (roi[2] * w)
            pts.append({**p, "x": round(float(nx), 5), "y": round(float(ny), 5)})
        out_labels[f["file"]] = {**L, "points": pts}
        if pts and not L["skip"]:
            a = np.array([[q["x"], q["y"]] for q in pts])
            spread.append(max(np.ptp(a[:, 0]), np.ptp(a[:, 1])))
    (DST / "frames.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=1))
    (ROOT / "data/real/labels_valc.json").write_text(json.dumps(out_labels, ensure_ascii=False, indent=1))
    rois = np.array([x["roi"][2] for x in out_meta])
    print(f"{len(out_meta)}장 → {DST}: 관심영역 한 변 중앙값 {np.median(rois):.2f}, 정답 점 퍼짐 중앙값 {np.median(spread):.2f}")


if __name__ == "__main__":
    main()

"""정답을 고친 뒤(13초 떨어뜨릴 때 돌아간 흰 한 번 반영) 실제 영상 회귀 결과를 모은다.

입력: reports/track_real_gt2_{pre,post,strict}_{Z1b,Z1,A,AB}_s{0..4}.json (scripts/track_video.py --segment ...)
      reports/track_real_gt2_strict_{B,Z2,all}_s{0..4}.json (기본을 늘 엄격으로 바꾼 뒤 다시 잰 것)
      B·Z2·all 의 pre·post 는 정답이 바뀌지 않아 reports/track_real_falseturn_summary.json 값을 그대로 옮긴다.
출력: reports/track_real_gt2_summary.json

설정
  pre    = 확정 규칙 전 (파이썬판 규칙)   --opts {"snapVsNone":-1e9,"snapOtherVsNone":-1e9}
  post   = 4점 / 8점 + 20프레임 예외 (정답 고치기 전의 기본)
  strict = 20프레임 예외 없이 늘 엄격 = 정답을 고친 뒤 바꾼 지금 앱 기본 (relaxWithin 0)
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
CFGS = ["pre", "post", "strict"]
OLD = {"pre": "전", "post": "후", "strict": "엄격만"}


def load(cfg: str, seg: str) -> list[dict]:
    return [json.loads((R / f"track_real_gt2_{cfg}_{seg}_s{s}.json").read_text()) for s in range(5)]


def acc(rs: list[dict]) -> dict:
    a = [r["move_acc"] for r in rs]
    return {"평균": round(st.mean(a), 3), "중앙값": round(st.median(a), 3), "표준편차": round(st.pstdev(a), 3),
            "시드별": [round(x, 3) for x in a], "판정_회전_수": [r["detected_moves"] for r in rs],
            "끝_상태_일치": sum(r["final_state_ok"] for r in rs)}


def false(rs: list[dict]) -> dict:
    f = [r["false_moves"] for r in rs]
    return {"합": sum(f), "시드별": f, "떨어뜨림_회전_찾음(시드별)": [r.get("true_found") for r in rs],
            "끝_상태_일치": sum(r["final_state_ok"] for r in rs)}


old = json.loads((R / "track_real_falseturn_summary.json").read_text())
out = {
    "설명": "sexy_move.mov 정답 수정: 섹시무브에 덧붙은 흰 한 번은 13초쯤 큐브를 떨어뜨렸다 다시 들 때 돌아간 것(촬영자 확인). "
            "그래서 A·AB 는 '처음 상태 + U'에서 시작하고 정답에서 그 한 수를 뺐으며, Z1·Z1b 의 정답은 그 한 수다. "
            "가짜 회전 = 판정한 회전 가운데 정답 회전과 짝짓지 못한 것(최장 공통 부분열). 시드 = 추적기 탐색 난수 0~4. "
            "B·Z2·all 의 pre·post 는 정답이 그대로라 track_real_falseturn_summary.json 에서 옮겼고, strict 는 기본을 바꾼 뒤 다시 쟀다.",
    "가짜_회전": {}, "정확도": {},
}
for seg in ["Z1b", "Z1"]:
    out["가짜_회전"][seg] = {c: false(load(c, seg)) for c in CFGS}
# B·Z2·all: pre·post 는 정답이 같아 예전 요약에서, strict(지금 기본)는 기본을 바꾼 뒤 다시 잰 파일에서
out["가짜_회전"]["Z2"] = {c: old["회전_없는_구간"]["Z2"][OLD[c]] for c in ["pre", "post"]} | {"strict": false(load("strict", "Z2"))}
for seg in ["A", "AB"]:
    out["정확도"][seg] = {c: acc(load(c, seg)) for c in CFGS}
for seg in ["B", "all"]:
    out["정확도"][seg] = {c: old["회전열_정확도"][seg][OLD[c]] for c in ["pre", "post"]} | {"strict": acc(load("strict", seg))}
(R / "track_real_gt2_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

for seg, v in out["가짜_회전"].items():
    print(seg, {c: (x["합"], x.get("시드별") or x.get("가짜_회전_수_시드별"), x.get("떨어뜨림_회전_찾음(시드별)")) for c, x in v.items()})
for seg, v in out["정확도"].items():
    print(seg, {c: (x.get("평균", x.get("정확도_평균")), x.get("중앙값"), x.get("시드별"), x.get("끝_상태_일치")) for c, x in v.items()})

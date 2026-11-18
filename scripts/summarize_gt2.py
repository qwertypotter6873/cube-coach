"""정답을 고친 뒤(13초 떨어뜨릴 때 돌아간 흰 한 번 반영) 실제 영상 회귀 결과를 모은다.

입력: reports/track_real_gt2_{pre,post,strict}_{Z1b,Z1,A,AB}_s{0..4}.json (scripts/track_video.py --segment ...)
      reports/track_real_gt2_strict_{B,Z2,all}_s{0..4}.json (기본을 늘 엄격으로 바꾼 뒤 다시 잰 것)
      B·Z2·all 의 pre·post 는 정답이 바뀌지 않아 reports/track_real_falseturn_summary.json 값을 그대로 옮긴다.
출력: reports/track_real_gt2_summary.json

설정
  pre    = 확정 규칙 전 (파이썬판 규칙)   --opts {"snapVsNone":-1e9,"snapOtherVsNone":-1e9}
  post   = 4점 / 8점 + 20프레임 예외 (정답 고치기 전의 기본)
  strict = 20프레임 예외 없이 늘 엄격 (relaxWithin 0) = 2026-11-18 전까지의 앱 기본
  gate   = strict + 자세 이탈 게이트 (offGate 1.0) = 2026-11-18부터 지금 앱 기본
           reports/track_real_gt2_gate_{seg}_s{0..4}.json
앱처럼 시작한 재기 (면 읽기가 끝나는 7.7초에 추적기를 만든다, 2026-11-18):
  들고 있기 7.7~19.5초, 시드 0~14: reports/track_real_pf_hold_{off,g10}_s*.json (off = 게이트 끔 = strict)
  끝까지 이어서 7.7~90초(app 구간), 시드 0~9: reports/track_real_gt2_app{off,gate}_s*.json
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
CFGS = ["pre", "post", "strict"]
NEW = "gate"
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
# 지금 기본(gate): 같은 구간·시드로 다시 잰 것
for seg in ["Z1b", "Z1", "Z2"]:
    out["가짜_회전"][seg][NEW] = false(load(NEW, seg))
for seg in ["A", "AB", "B", "all"]:
    out["정확도"][seg][NEW] = acc(load(NEW, seg))

# 앱처럼 시작한 재기: 8.2초에 새 추적기를 만드는 Z1b 는 유난히 유리한 시작이었다 (시작 프레임에 따라 평균 3.6~6.8)
def hold(cfg: str) -> dict:
    raw, drop = [], []
    for s in range(15):
        ev = [e for e in json.loads((R / f"track_real_pf_hold_{cfg}_s{s}.json").read_text())["events"] if 8.2 <= e[2] <= 19.5]
        raw.append(len(ev))
        drop.append(any(e[0] == 0 and e[1] == 1 and 12.5 <= e[2] <= 15.0 for e in ev))
    return {"판정_수_평균": round(st.mean(raw), 2), "범위": [min(raw), max(raw)], "시드별": raw, "떨어뜨림_회전_찾음": sum(drop)}


out["앱처럼_시작_들고_있기(7.7~19.5초, 8.2~19.5초 판정 수, 시드 15개)"] = {"strict": hold("off"), NEW: hold("g10")}
out["앱처럼_시작_끝까지(7.7~90초, 시드 10개)"] = {c: acc([json.loads((R / f"track_real_gt2_app{t}_s{s}.json").read_text()) for s in range(10)])
                                                for c, t in (("strict", "off"), (NEW, "gate"))}
(R / "track_real_gt2_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

for seg, v in out["가짜_회전"].items():
    print(seg, {c: (x["합"], x.get("시드별") or x.get("가짜_회전_수_시드별"), x.get("떨어뜨림_회전_찾음(시드별)")) for c, x in v.items()})
for seg, v in out["정확도"].items():
    print(seg, {c: (x.get("평균", x.get("정확도_평균")), x.get("중앙값"), x.get("시드별"), x.get("끝_상태_일치")) for c, x in v.items()})

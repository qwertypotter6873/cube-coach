"""네 번째 시험 영상(latest3.mov, 150초까지) 결과를 모은다. 입력 192 대 256, 게이트 켬·끔을 가르려고 찍은 영상이다.

판정 규칙 (2026-11-19, 추적기를 돌리기 전에 정답과 함께 커밋):
  갈래   gate = 지금 앱 기본 (입력 192, 자세 이탈 게이트 1.0)          --tag latest3_gate_<구간>_s<시드>
         off  = 게이트 끔 (입력 192)                                 --opts {"offGate":0}
         r256 = 추적 단계 입력 256, 지도는 넓이 평균으로 48x48 (게이트 켬)  --res 256 --map-n 48
  구간   코칭 구간 R·L·F (구간 시작 상태를 알려 줌, 각 24수, 한 수씩) 과 앱 흐름 afterscan(앱이 처음 상태를 읽은 16.2초부터
         146.5초까지 이어서). 시드 0~14.
  1. 입력 256: 코칭 구간 R·L·F x 시드 15개의 회전열 정확도를 (구간, 시드) 짝 45개로 견줘 (r256 − gate) 평균의
     부트스트랩 95% 구간 아래 끝이 0보다 크고, afterscan 에서 동작 창 안 정답과 짝지은 회전 수의 평균 차이가 0 이상이면
     넣는다. 아니면 192 그대로.
  2. 게이트: 켠 채로 둔다. off 가 1과 같은 기준(코칭 구간 짝 45개, 구간 아래 끝 > 0)으로 나으면 사용자에게 알린다.
  3. 보고: 구간별 정확도(평균·중앙값·24수를 모두 맞힌 시드·끝 상태), afterscan 의 동작 창 안 짝·창 밖 판정·회전 없는
     구간의 가짜 회전, 게이트가 막은 회전 수(off_rejected).
  4. 이 영상으로 설정을 고치지 않는다.
입력: reports/track_real_latest3_{gate,off,r256}_{R,L,F,afterscan}_s{0..14}.json
출력: reports/track_real_latest3_summary.json
실행: .venv/bin/python scripts/summarize_latest3.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import LATEST3_F, LATEST3_L, LATEST3_R, lcs  # noqa: E402

R = ROOT / "reports"
ARMS = ["gate", "off", "r256"]
SEGS = ["R", "L", "F"]
SEEDS = range(15)
WIN = {"R": (66.5, 85.0, LATEST3_R), "L": (91.5, 107.5, LATEST3_L), "F": (112.6, 146.5, LATEST3_F)}  # 첫 회전 0.7초 전 ~ 끝 0.7초 뒤
QUIET = [(16.2, 66.5), (85.0, 91.5), (107.5, 112.6)]  # afterscan 안의 회전 없는 구간 (면 보여 주기 끝·들고 있기, 두 번 멈춤)


def load(arm: str, seg: str, s: int) -> dict:
    return json.loads((R / f"track_real_latest3_{arm}_{seg}_s{s}.json").read_text())


def boot(d: np.ndarray, n: int = 4000) -> list[float]:
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> None:
    out: dict = {"설명": __doc__.strip(), "코칭_구간": {}, "앱_흐름": {}, "판정": {}}
    acc = {a: {g: np.array([load(a, g, s)["move_acc"] for s in SEEDS]) for g in SEGS} for a in ARMS}
    for g in SEGS:
        out["코칭_구간"][g] = {}
        for a in ARMS:
            rs = [load(a, g, s) for s in SEEDS]
            x = acc[a][g]
            out["코칭_구간"][g][a] = {"평균": round(float(x.mean()), 3), "중앙값": round(float(np.median(x)), 3),
                                    "24수_전부_맞힌_시드": int((x >= 0.999).sum()), "끝_상태_일치": sum(bool(r["final_state_ok"]) for r in rs),
                                    "시드별": [round(float(v), 3) for v in x], "게이트가_막은_회전": sum(r.get("off_rejected", 0) for r in rs),
                                    "입력": sorted({(r.get("res"), r.get("map_n")) for r in rs})}
    flow = {}
    for a in ARMS:
        rows = []
        for s in SEEDS:
            r = load(a, "afterscan", s)
            ev = [(e[0], e[1], e[2]) for e in r["events"]]
            ins = {k: [(c, q) for c, q, t in ev if w[0] <= t <= w[1]] for k, w in WIN.items()}
            matched = {k: lcs(ins[k], WIN[k][2]) for k in WIN}
            rows.append({"짝": sum(matched.values()), "구간별_짝": matched, "창_밖": len(ev) - sum(len(v) for v in ins.values()),
                         "회전_없는_구간_판정": sum(1 for c, q, t in ev if any(a0 <= t <= a1 for a0, a1 in QUIET)),
                         "정확도": r["move_acc"], "끝_상태": bool(r["final_state_ok"]), "게이트가_막은_회전": r.get("off_rejected", 0)})
        flow[a] = rows
        out["앱_흐름"][a] = {"창_안_정답과_짝지은_회전(72수 중, 시드별)": [x["짝"] for x in rows],
                           "평균": round(st.mean(x["짝"] for x in rows), 2),
                           "구간별_짝_평균": {k: round(st.mean(x["구간별_짝"][k] for x in rows), 2) for k in WIN},
                           "72수_전부_맞힌_시드": sum(x["짝"] == 72 for x in rows),
                           "창_밖_판정_합": sum(x["창_밖"] for x in rows), "회전_없는_구간_판정_합": sum(x["회전_없는_구간_판정"] for x in rows),
                           "끝_상태_일치": sum(x["끝_상태"] for x in rows), "게이트가_막은_회전_합": sum(x["게이트가_막은_회전"] for x in rows)}
    for a in ("r256", "off"):
        d = np.concatenate([acc[a][g] - acc["gate"][g] for g in SEGS])
        lo_hi = boot(d)
        fm = np.array([x["짝"] for x in flow[a]], float) - np.array([x["짝"] for x in flow["gate"]], float)
        out["판정"][f"{a}_대_gate"] = {"코칭_구간_정확도_차이_평균": round(float(d.mean()), 3), "95%": lo_hi,
                                     "나은_짝": int((d > 0).sum()), "같은_짝": int((d == 0).sum()), "못한_짝": int((d < 0).sum()),
                                     "앱_흐름_짝_차이_평균": round(float(fm.mean()), 2), "앱_흐름_95%": boot(fm),
                                     "규칙_충족": bool(lo_hi[0] > 0 and fm.mean() >= 0)}
    out["판정"]["결론"] = {"입력_256을_넣는가": out["판정"]["r256_대_gate"]["규칙_충족"],
                         "게이트를_끈_쪽이_나은가(알릴_일)": bool(out["판정"]["off_대_gate"]["95%"][0] > 0)}
    (R / "track_real_latest3_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for g in SEGS:
        print(g, {a: (v["평균"], v["중앙값"], v["24수_전부_맞힌_시드"], v["끝_상태_일치"]) for a, v in out["코칭_구간"][g].items()})
    for a, v in out["앱_흐름"].items():
        print("앱 흐름", a, v["평균"], v["72수_전부_맞힌_시드"], v["창_밖_판정_합"], v["회전_없는_구간_판정_합"], v["구간별_짝_평균"])
    print(json.dumps(out["판정"], ensure_ascii=False))


if __name__ == "__main__":
    main()

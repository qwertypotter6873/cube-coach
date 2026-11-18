"""문턱을 고른 뒤 찍은 시험 영상(latest.mov, latest2.mov) 결과를 모은다. 문턱은 sexy_move.mov 로 정한 그대로 두고 시험만 했다.

실행: .venv/bin/python scripts/summarize_latest.py [latest|latest2]   (기본 latest)
입력: reports/track_real_<영상>_{cfg}_{seg}_s{0..4}.json (scripts/track_video.py --segment <영상>_<seg>)
출력: reports/track_real_<영상>_summary.json

설정 (추적기를 돌리기 전에 고정)
  strict = 지금 앱 기본 (4점 / 8점, 예외 없음)         ← 대표 값
  pre    = 확정 규칙 전 (파이썬판 규칙)                  --opts {"snapVsNone":-1e9,"snapOtherVsNone":-1e9}
  post   = 4점 / 8점 + 20프레임 예외 (지난번 기본)      --opts {"relaxWithin":20}
구간: scripts/track_video.py 의 LATEST_GT, LATEST2_GT
"""

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from track_video import LATEST2_L, LATEST2_R, LATEST_L, LATEST_R, VIDEO_GT, lcs  # noqa: E402

VID = sys.argv[1] if len(sys.argv) > 1 else "latest"
GT = VIDEO_GT[VID][1]
CONF = {
    "latest": {"설명": "latest.mov(194초, 노트북 웹캠, 빠르게 돌림)", "정확도": ["R", "L", "ZR", "all"],
               "창": {"R": (52.5, 72.8, LATEST_R), "L": (85.0, 96.8, LATEST_L)}, "이어서": [("ZR", ["R"]), ("all", ["R", "L"])]},
    "latest2": {"설명": "latest2.mov(180초까지, 노트북 웹캠, 한 수씩 천천히)", "정확도": ["R", "L", "RL", "all"],
                "창": {"R": (64.0, 97.5, LATEST2_R), "L": (104.0, 134.5, LATEST2_L)}, "이어서": [("RL", ["R", "L"]), ("all", ["R", "L"])]},
}[VID]

R = ROOT / "reports"
CFGS = ["strict", "pre", "post"]
SEEDS = range(5)


def load(cfg: str, seg: str) -> list[dict]:
    return [json.loads((R / f"track_real_{VID}_{cfg}_{seg}_s{s}.json").read_text()) for s in SEEDS]


def acc(rs: list[dict], seg: str) -> dict:
    a = [r["move_acc"] for r in rs]
    out = {"중앙값": round(st.median(a), 3), "평균": round(st.mean(a), 3), "표준편차": round(st.pstdev(a), 3),
           "시드별": [round(x, 3) for x in a], "판정_회전_수(같은 면 합침)": [r["detected_moves"] for r in rs],
           "정답_회전_수": rs[0]["expected_moves"], "끝_상태_일치": sum(bool(r["final_state_ok"]) for r in rs),
           "처음_잡은_시각": [r["first_lock_t"] for r in rs]}
    if VID == "latest" and seg == "R":  # 첫 R U R' U'(52.7~53.2초, 0.6초에 4수)를 몇 수 잡았나
        out["첫_4수_가운데_잡은_수"] = [lcs([(e[0], e[1]) for e in r["events"] if e[2] < 53.6], LATEST_R[:4]) for r in rs]
    return out


def false(rs: list[dict]) -> dict:
    f = [r["false_moves"] for r in rs]
    return {"합": sum(f), "시드별": f, "중앙값": st.median(f), "초": rs[0]["seconds"],
            "끝_상태_일치": sum(bool(r["final_state_ok"]) for r in rs)}


out = {"설명": CONF["설명"] + " 시험 결과. 문턱은 sexy_move.mov 로 정한 그대로다. 시드 = 추적기 탐색 난수 0~4. "
               "정답 구간과 처음 상태는 추적기를 돌리기 전에 프레임만 보고 정했다(scripts/track_video.py LATEST_GT·LATEST2_GT).",
       "구간": {k: {"시작": v[0], "끝": v[1], "정답_회전_수": None if v[2] is None else len(v[2])} for k, v in GT.items()},
       "정확도": {}, "가짜_회전": {}, "정답_없는_구간": {}}
for seg in CONF["정확도"]:
    out["정확도"][seg] = {c: acc(load(c, seg), seg) for c in CFGS}
for seg in ["Z1", "Z1b", "Z2", "Z3"]:
    out["가짜_회전"][seg] = {c: false(load(c, seg)) for c in CFGS}
fr = load("strict", "free")
out["정답_없는_구간"][f"free({GT['free'][0]:.0f}~{fr[0]['seconds'] + GT['free'][0]:.0f}초, strict)"] = {
    "판정_회전_수": [r["detected_moves"] for r in fr], "추적_비율": [round(r["tracked_ratio"], 3) for r in fr],
    "놓친_횟수": [r["lost_episodes"] for r in fr], "초": fr[0]["seconds"]}
# 이어서 잰 구간은 편집 거리 정확도가 부푼다(가짜가 우연히 정답과 겹침). 동작 구간 안 판정만 정답과 짝짓고, 밖 판정은 따로 센다.
WIN = CONF["창"]
for seg, parts in CONF["이어서"]:
    out["정확도"][seg + "_제자리"] = {}
    for c in CFGS:
        rs = load(c, seg)
        match, outside = [], []
        for r in rs:
            ev = [(e[0], e[1], e[2]) for e in r["events"]]
            inside = 0
            m = 0
            for k in parts:
                t0, t1, g = WIN[k]
                w = [(a, b) for a, b, t in ev if t0 <= t <= t1]
                inside += len(w)
                m += lcs(w, g)
            match.append(m)
            outside.append(len(ev) - inside)
        out["정확도"][seg + "_제자리"][c] = {"동작_구간_안_정답과_짝지은_회전(시드별)": match, "동작_구간_밖_판정(시드별)": outside,
                                          "밖_판정_합": sum(outside)}


def matched_events(events: list, g: list) -> list:
    """판정과 정답의 최장 공통 부분열에서 짝지어진 판정."""
    a = [(e[0], e[1]) for e in events]
    n, m = len(a), len(g)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            D[i + 1][j + 1] = D[i][j] + 1 if a[i] == g[j] else max(D[i][j + 1], D[i + 1][j])
    i, j, got = n, m, []
    while i and j:
        if a[i - 1] == g[j - 1] and D[i][j] == D[i - 1][j - 1] + 1:
            got.append(events[i - 1])
            i, j = i - 1, j - 1
        elif D[i - 1][j] >= D[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return got


# 확정 규칙 전(pre) 설정에서 정답과 짝지은 판정 가운데 '안 돌았다'보다 4점 미만으로 나은 것 = 지금 규칙이면 걸러질 진짜 회전
out["확정_규칙이_거를_진짜_회전"] = {}
for seg, g in [(k, v[2]) for k, v in WIN.items()]:
    per = []
    for r in load("pre", seg):
        got = matched_events(r["events"], g)
        per.append([len(got), sum(1 for e in got if len(e) > 3 and e[3].get("vsNone", 99) < 4)])
    out["확정_규칙이_거를_진짜_회전"][seg] = {"시드별_[짝지은_판정, 그_가운데_4점_미만]": per}
if VID == "latest2":  # 결과를 본 뒤 더한 구간 (지금 기본만): 두 동작 끝까지 이어서
    for seg, name in [("app", "app(51~134.5초, 면 보여 주기를 마친 뒤부터, 결과를 본 뒤 더함, strict만)"),
                      ("afterscan", "afterscan(32.86~134.5초, 면 읽기를 고친 뒤 앱이 읽기를 마치는 시각부터, 결과를 본 뒤 더함, strict만)")]:
        rs = load("strict", seg)
        out["정확도"][name] = acc(rs, seg)
        m, o = [], []
        for r in rs:
            ev = [(e[0], e[1], e[2]) for e in r["events"]]
            ins = [[(a, b) for a, b, t in ev if WIN[k][0] <= t <= WIN[k][1]] for k in ("R", "L")]
            m.append(lcs(ins[0], WIN["R"][2]) + lcs(ins[1], WIN["L"][2]))
            o.append(len(ev) - len(ins[0]) - len(ins[1]))
        out["정확도"][name].update({"동작_구간_안_정답과_짝지은_회전(시드별)": m, "동작_구간_밖_판정(시드별)": o})
(R / f"track_real_{VID}_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

for seg, v in out["정확도"].items():
    if seg.endswith("_제자리") or seg.startswith(("app", "afterscan")):
        continue
    print(seg, {c: (x["중앙값"], x["평균"], x["시드별"], x["판정_회전_수(같은 면 합침)"], x["끝_상태_일치"],
                    x.get("첫_4수_가운데_잡은_수")) for c, x in v.items()})
for seg, v in out["가짜_회전"].items():
    print(seg, {c: (x["합"], x["시드별"]) for c, x in v.items()})
print(out["정답_없는_구간"])
print({k: v for k, v in out["정확도"].items() if k.endswith("_제자리")})
print(out["확정_규칙이_거를_진짜_회전"])

#!/bin/zsh
# 배포용 빌드(vite preview, 127.0.0.1:5182)에서 샘플 영상(46초)을 처음부터 돌리고, 앱이 받은 회전(시각, 가운데 색, 90도 단위)을 출력한다.
# 결과 정리: reports/browser_sample_runs.json
#
# 사람이 쓰는 길 그대로 연다: '카메라 · 영상' 고르기 → '샘플 영상' 버튼 (그래야 코치가 읽은 상태로 시작한다).
# 끝에서 화면·코치가 아는 큐브가 추적기 상태와 같은지, 읽은 처음 상태가 정답과 같은지 확인한다 (놓친 회전을 앞에 끼워 넣어도 따라가야 한다).
#   표준출력 마지막 줄: 예전과 같은 결과 JSON (phase, scanned, n, moves=[시각, 가운데 색, 90도 단위], errs, fps + 확인용 값, scanT=상태를 다 읽은 영상 시각, frames/skipped=처리한·건너뛴 프레임 수,
#     ms=프레임마다 걸린 시간의 중앙값·90% 값: 모델(자르기·추론·지도 줄이기 포함)·추적기·그리기, 최근 2000프레임.
#        firstTrackModel=추적으로 넘어간 첫 프레임(입력 192→256)의 모델 시간, modelMax=가장 느린 모델 시간, maxGap=처리한 프레임 사이 가장 긴 간격(ms), gaps200=200ms 넘는 간격 수)
#   표준오류: 확인 결과. 같지 않으면 종료 코드 1.
# 다른 주소·세션: CUBE_URL=http://127.0.0.1:5180/ AGENT_BROWSER_SESSION=내세션 scripts/browser_sample_check.sh
export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION:-cubesmoke-main}"
URL="${CUBE_URL:-http://127.0.0.1:5182/}"
ROOT="${0:A:h:h}"
AB=agent-browser
fail() { print -u2 -- "확인 실패: $1"; $AB close >/dev/null 2>&1; exit 1; }

$AB open "$URL" >/dev/null
for i in $(seq 1 30); do sleep 1; R=$($AB eval "!!globalThis.__vision" 2>/dev/null | tr -d "\n "); [ "$R" = "true" ] && break; done
# 회전 이벤트를 받는 쪽(main.ts)에 넘기기 전에 기록만 한다. 인자는 모두 그대로 넘긴다 (둘째 인자 = 추적기 상태).
$AB eval "(() => { const v = globalThis.__vision; globalThis.__moves = []; globalThis.__backs = []; globalThis.__errs = []; globalThis.__scanT = null;
  const os = v.ev.scanned.bind(v.ev); v.ev.scanned = (...a) => { globalThis.__scanT = +v.video.currentTime.toFixed(2); return os(...a); };
  addEventListener('error', e => globalThis.__errs.push(String(e.message)));
  const orig = v.ev.move.bind(v.ev);
  v.ev.move = (...a) => { const [m, sync] = a; const t = +v.video.currentTime.toFixed(2);
    globalThis.__moves.push([t, m.color, m.k]); if (sync && sync.back > 0) globalThis.__backs.push([t, m.color, m.k, sync.back]);
    return orig(...a); };
  return 'ok'; })()" >/dev/null
$AB click "#modeCam" >/dev/null || fail "'카메라 · 영상' 버튼을 누르지 못했어요"
M=$($AB eval "document.body.dataset.mode" 2>/dev/null | tr -d '\n "')
[ "$M" = "camera" ] || fail "카메라 모드가 아니에요 (mode=$M)"
$AB click "#camSample" >/dev/null || fail "'샘플 영상' 버튼을 누르지 못했어요"
for i in $(seq 1 60); do sleep 2; E=$($AB eval "!!(globalThis.__vision.video && globalThis.__vision.video.ended)" 2>/dev/null | tr -d '\n '); [ "$E" = "true" ] && break; done
sleep 1 # 끝나기 직전 프레임의 처리가 마저 끝나게
OUT=$($AB eval "JSON.stringify({phase: globalThis.__vision.phase, scanned: document.body.dataset.scanned, scanT: globalThis.__scanT, n: globalThis.__moves.length, moves: globalThis.__moves, errs: globalThis.__errs, fps: globalThis.__vision.fps,
  mode: document.body.dataset.mode, level: document.getElementById('level').value, backs: globalThis.__backs, frames: globalThis.__vision.frames, skipped: globalThis.__vision.skipped,
  cube: document.body.dataset.cube || '', coach: document.body.dataset.coach || '', tracker: Array.from((globalThis.__vision.tracker || {}).colors || []).join(''),
  ms: (() => { const st = globalThis.__vision.stats || []; const q = (k, p) => { const a = st.map(x => x[k]).sort((x, y) => x - y); return a.length ? +a[Math.min(a.length - 1, Math.floor(p * a.length))].toFixed(2) : null; };
    const ft = st.find(x => x.kind !== 'scan'); let mx = null, gap = 0, gapAt = null, gaps = 0;
    for (let i = 0; i < st.length; i++) { if (!mx || st[i].modelMs > mx.modelMs) mx = st[i]; if (i) { const g = st[i].t - st[i - 1].t; if (g > 200) gaps++; if (g > gap) { gap = g; gapAt = st[i].kind; } } }
    return {n: st.length, model50: q('modelMs', 0.5), model90: q('modelMs', 0.9), track50: q('trackMs', 0.5), track90: q('trackMs', 0.9), draw50: q('drawMs', 0.5),
      firstTrackModel: ft ? +ft.modelMs.toFixed(1) : null, modelMax: mx ? +mx.modelMs.toFixed(1) : null, modelMaxKind: mx ? mx.kind : null, maxGap: gap, maxGapKind: gapAt, gaps200: gaps}; })()})")
$AB close >/dev/null 2>&1
print -r -- "$OUT"

# 확인: 화면(main.ts cube)과 코치가 아는 큐브가 추적기 상태와 같은가.
# 추적기 배열은 '층을 잡고 몸통을 돌린' 회전 때문에 통째로 돌아가 있을 수 있어, 가운데 칸 자리를 맞춘 뒤 견준다 (web/src/coach/resync.ts).
print -r -- "$OUT" | node --input-type=module -e '
import { pathToFileURL } from "node:url";
const { alignCenters } = await import(pathToFileURL(process.argv[1] + "/web/src/coach/resync.ts").href);
let raw = ""; for await (const c of process.stdin) raw += c;
const r = JSON.parse(JSON.parse(raw.trim()));
const st = (s) => Uint8Array.from(s, Number);
const problems = [];
if (r.mode !== "camera") problems.push(`카메라 모드가 아님 (${r.mode})`);
if (!r.scanned) problems.push("상태를 읽지 못함");
// 샘플 영상의 처음 상태(사람이 읽은 정답, scripts/track_video.py SEXY_MOVE_STATE 를 앱 숫자로). 상태 일치만 보면
// 틀리게 읽은 상태도 서로 같기만 하면 통과한다(2026-11-18, 31.69초에 18칸 틀린 실행이 '통과'로 나왔다).
const TRUTH = "211300354513010150440524250432330341112441420523252535";
if (r.scanned && r.scanned !== TRUTH) problems.push(`읽은 처음 상태가 정답과 다름 (${[...r.scanned].filter((c, i) => c !== TRUTH[i]).length}칸, ${r.scanT}초에 읽음)`);
if (r.tracker.length !== 54 || r.cube.length !== 54) problems.push("추적기 또는 화면 상태가 없음");
else {
  const tracker = Array.from(alignCenters(st(r.tracker), st(r.cube))).join("");
  if (r.cube !== tracker) problems.push(`화면 상태 ≠ 추적기 상태 (${[...r.cube].filter((c, i) => c !== tracker[i]).length}칸 다름)`);
  if (r.coach !== r.cube) problems.push("코치 상태 ≠ 화면 상태");
}
const note = `회전 ${r.n}개, 앞에 끼워 넣은 회전 ${r.backs.length}개${r.backs.length ? "" : " (이번 실행에서는 끼워 넣기가 없어 다시 맞추기는 시험되지 않음)"}`;
if (problems.length) { console.error(`확인 실패: ${problems.join(", ")}. ${note}`); process.exit(1); }
console.error(`확인 통과: 읽은 처음 상태 = 정답, 화면·코치 상태 = 추적기 상태. ${note}`);
' "$ROOT"

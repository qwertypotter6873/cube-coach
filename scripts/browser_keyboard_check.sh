#!/bin/zsh
# 가상 큐브(키보드) 모드를 사람처럼 눌러 보는 확인. 배포용 빌드(기본 127.0.0.1:5182) 또는 CUBE_URL 의 공개 주소에서 돈다.
#   1. 따라 하기: 지시와 다른 면을 돌리면 바로잡기 카드가 뜨고, 되돌리면 원래 지시로 돌아오는가
#   3. 그 풀이를 다음 할 일 카드의 기호대로 끝까지 따라 하면 기록·완성 정리가 1개씩인가 (잡는 법 카드는 화살표로 24방향을 훑는다)
#   4. 다 맞춘 뒤 한 수 돌렸다 되돌려도 기록이 늘지 않는가
#   2. 힌트만: 새 페이지에서 시작할 때 다음 동작·공식 기호를 보이지 않는가
# 결과는 단계마다 JSON 한 줄 (errs = 페이지 오류). 판정은 사람이 읽는다.
HELPER=$(cat <<'JS'
(() => {
  const KEY = { R: ["i", "k"], L: ["d", "e"], U: ["j", "f"], D: ["s", "l"], F: ["h", "g"], B: ["w", "o"] };
  const WALK = "LLLULLLULLLDLLLDLLLULLL";
  const ARROW = { L: "ArrowLeft", U: "ArrowUp", D: "ArrowDown" };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const tap = async (key, hold = 360) => {
    dispatchEvent(new KeyboardEvent("keydown", { key }));
    await sleep(hold);
    dispatchEvent(new KeyboardEvent("keyup", { key }));
    await sleep(160);
  };
  const card = () => ({ state: document.getElementById("next").dataset.state, sym: document.getElementById("nextSym").textContent.trim() });
  const press = async (sym) => {
    const m = /^([RLUDFB])(2|')?$/.exec(sym);
    if (!m) return false;
    const [ccw, cw] = [KEY[m[1]][1], KEY[m[1]][0]];
    if (m[2] === "2") { await tap(cw); await tap(cw); }
    else await tap(m[2] === "'" ? ccw : cw);
    return true;
  };
  window.__kb = {
    errs: [],
    walk: 0,
    card,
    press,
    tap,
    sleep,
    history: () => { try { return JSON.parse(localStorage.getItem("cubecoach.history.v1") || "[]").length; } catch { return -1; } },
    summaries: () => document.querySelectorAll("#log li.summary").length,
    /** 한 걸음: 잡는 법 카드면 화살표 한 번, 동작 카드면 그 기호대로 돌린다 */
    async step() {
      const c = card();
      if (c.state === "done") return "done";
      if (c.state === "hold") {
        if (this.walk >= WALK.length) return "hold-stuck";
        await tap(ARROW[WALK[this.walk++]], 60);
        await sleep(250);
        return "hold";
      }
      this.walk = 0;
      if (c.state === "move" || c.state === "fix") {
        const ok = await press(c.sym);
        await sleep(200);
        return ok ? `moved ${c.sym}` : `unknown ${c.sym}`;
      }
      await sleep(250);
      return `wait ${c.state}`;
    },
  };
  addEventListener("error", (e) => window.__kb.errs.push(String(e.message)));
  addEventListener("unhandledrejection", (e) => window.__kb.errs.push(String(e.reason)));
  return "ok";
})()
JS
)
export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION:-kbsmoke}"
AB=agent-browser
URL=${CUBE_URL:-http://127.0.0.1:5182/}
$AB open "$URL" >/dev/null
for i in $(seq 1 40); do sleep 1; R=$($AB eval "!!globalThis.__vision" 2>/dev/null | tr -d "\n "); [ "$R" = "true" ] && break; done
$AB eval "$HELPER" >/dev/null
$AB eval "localStorage.removeItem('cubecoach.history.v1'); 'ok'" >/dev/null
$AB click "#modeKey" >/dev/null
setsel() { $AB eval "(() => { const s = document.getElementById('$1'); s.value = '$2'; s.dispatchEvent(new Event('change', {bubbles: true})); return s.value; })()" >/dev/null; }
setsel method beginner; setsel level guided
$AB click "#scramble" >/dev/null; sleep 1
echo "== 1. 따라 하기: 틀린 면 → 되돌리기"
$AB eval "(async () => { for (let i = 0; i < 30 && __kb.card().state === 'hold'; i++) await __kb.step();
  const before = __kb.card(); if (before.state !== 'move') return JSON.stringify({why: 'no move card', before});
  const wrong = ['R','L','U','D','F','B'].find(l => l !== before.sym[0]);
  await __kb.press(wrong); await __kb.sleep(800); const during = __kb.card();
  await __kb.press(wrong + \"'\"); await __kb.sleep(1000); const after = __kb.card();
  return JSON.stringify({before, wrong, during, after, pass: during.state === 'fix' && after.state === 'move' && after.sym === before.sym, errs: __kb.errs}); })()"
echo "== 3. 따라 하기로 끝까지 풀기 (1번 세션을 이어서)"
$AB eval "localStorage.removeItem('cubecoach.history.v1'); window.__n = 0; window.__last = ''; 'ok'" >/dev/null
for chunk in $(seq 1 40); do
  R=$($AB eval "(async () => { for (let i = 0; i < 25; i++) { const r = await __kb.step(); window.__last = r; if (r.startsWith('moved')) window.__n++; if (r === 'done' || r.startsWith('unknown') || r === 'hold-stuck') break; } return window.__last; })()" | tr -d '"\n')
  case "$R" in done|unknown*|hold-stuck) break;; esac
done
$AB eval "JSON.stringify({last: window.__last, moves: window.__n, card: __kb.card(), history: __kb.history(), summaries: __kb.summaries(), errs: __kb.errs})"
echo "== 4. 다 맞춘 뒤 한 수 돌렸다 되돌리기: 기록이 늘지 않는다"
$AB eval "(async () => { await __kb.press('R'); await __kb.sleep(600); await __kb.press(\"R'\"); await __kb.sleep(1200); return JSON.stringify({card: __kb.card(), history: __kb.history(), summaries: __kb.summaries(), errs: __kb.errs}); })()"
$AB close >/dev/null 2>&1
echo "== 2. 힌트만: 새 페이지에서 시작할 때 공식·다음 동작을 보이지 않는다"
export AGENT_BROWSER_SESSION="${AGENT_BROWSER_SESSION}-2" # 닫은 세션을 다시 열면 페이지가 붙지 않아 새 세션으로 연다
$AB open "$URL" >/dev/null
for i in $(seq 1 60); do sleep 1; R=$($AB eval "!!globalThis.__vision && !!document.getElementById('modeKey')" 2>/dev/null | tr -d "\n "); [ "$R" = "true" ] && break; done
$AB eval "globalThis.__errs = []; addEventListener('error', e => __errs.push(String(e.message))); 'ok'" >/dev/null
$AB click "#modeKey" >/dev/null
setsel method beginner; setsel level hint
$AB click "#scramble" >/dev/null; sleep 1.5
$AB eval "JSON.stringify({state: document.getElementById('next').dataset.state, badge: document.getElementById('nextBadge').textContent, sym: document.getElementById('nextSym').textContent, alg: !!document.querySelector('#goal code.alg'), errs: __errs})"
$AB close >/dev/null 2>&1

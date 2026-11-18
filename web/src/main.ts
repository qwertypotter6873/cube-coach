/**
 * 화면 조립: 입력(카메라 추적기 또는 키보드 가상 큐브) → 코치 → 음성 → 화면.
 *
 * 화면은 늘 한 곳만 보게 한다.
 *   - 처음: 입력과 설정을 고르는 화면뿐.
 *   - 카메라 모드: 손에 든 진짜 큐브(카메라 화면)가 주 화면, 바로 아래 '다음 할 일' 카드.
 *     가상 큐브는 옆 칸의 "코치가 아는 큐브"로 접어 둔다.
 *   - 키보드 모드: 가상 큐브가 주 화면, 바로 아래 같은 카드.
 * 지금 어디쯤인지는 단계 띠(stepper)와 한 문장 안내(todo)로 늘 보여 준다.
 *
 * 가상 큐브는 카메라 추적기 자리를 대신한다. 추적기와 똑같이 세 가지 이벤트를 코치에 보낸다.
 *   - 층이 돌아가는 중 (매 프레임, 각도)  → coach.onTurning
 *   - 회전 하나가 끝남                    → coach.onMove
 *   - 큐브를 쥔 방향이 바뀜               → coach.onPose
 */

import type { ColorMove, Letter, State, View } from "./coach/cube.ts";
import { COLOR_KO, SOLVED, applySlotAlg, randomScramble, rotateView, viewColors } from "./coach/cube.ts";
import type { Level, Utterance } from "./coach/coach.ts";
import { Coach } from "./coach/coach.ts";
import type { Method } from "./coach/method.ts";
import { STAGES, analyze, stageOf, stageProgress } from "./coach/method.ts";
import { STAGE_NAME, moveLetter, moveSymbol, moveText } from "./coach/speech.ts";
import { ClipEngine, VoiceQueue, WebSpeechEngine } from "./coach/voice.ts";
import { caseStats, compareStages, feedbackText, loadHistory, saveSolve } from "./coach/progress.ts";
import { Drill } from "./coach/drill.ts";
import { OCLL, PLL } from "./coach/method.ts";
import type { SolveRecord } from "./coach/progress.ts";
import type { SpeechEngine } from "./coach/voice.ts";
import { HEX, renderCube, renderHidden, renderMoveArrow } from "./ui/cubeView.ts";
import { VisionInput } from "./vision/camera.ts";
import type { MoveSync, VisionEvents } from "./vision/camera.ts";
import { followMove } from "./coach/resync.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** 같은 글자면 다시 쓰지 않는다 (매 프레임 그리므로) */
function setText(el: HTMLElement, s: string): void {
  if (el.textContent !== s) el.textContent = s;
}
const htmlCache = new WeakMap<HTMLElement, string>();
function setHtml(el: HTMLElement, s: string): void {
  if (htmlCache.get(el) === s) return;
  htmlCache.set(el, s);
  el.innerHTML = s;
}
function show(el: HTMLElement, on: boolean): void {
  if (el.hidden === on) el.hidden = !on;
}
function setData(el: HTMLElement, key: string, v: string): void {
  if (el.dataset[key] !== v) el.dataset[key] = v;
}
/** 은/는 같은 받침 조사 대신, 색 이름 뒤에는 "이/가"만 쓴다 */
const iGa = (w: string) => {
  const c = w.charCodeAt(w.length - 1);
  return w + (c >= 0xac00 && c <= 0xd7a3 && (c - 0xac00) % 28 ? "이" : "가");
};

// --- 상태 ---------------------------------------------------------------------

let cube: State = SOLVED;
let pose: View = { up: 3, front: 2 };
let coach: Coach | null = null;
let drill: Drill | null = null;
const web = new WebSpeechEngine();
const voice = new VoiceQueue(web);

type InputMode = "keyboard" | "camera";
let mode: InputMode = "keyboard";
let picked = false; // 입력을 골랐는가 (단계 띠 ①)
let scanned: State | null = null; // 카메라로 읽은 지금 큐브 상태
let scanLeft: number[] = []; // 아직 못 읽은 면의 가운데 색
let scanCount: number[] = [0, 0, 0, 0, 0, 0]; // 면마다 읽은 정도 (0~8)
let lastKind = ""; // 코치가 마지막으로 한 말의 종류 (바로잡기 구분용)
let lastGoal = ""; // 코치가 마지막으로 말한 목표 (한 문장 안내에 쓴다)
let revealAt = -1; // 힌트만 수준: 이 수 번호에서 코치가 동작을 불러 줬다
let trail: string[] = []; // 최근에 돌린 회전 (혼자 풀기에서 보여 준다)
let camErr = ""; // 카메라·영상 열기 실패 안내
let camSource: "camera" | "sample" | "file" | null = null; // 지금 보고 있는 영상
let everTracked = false; // 이번 영상에서 한 번이라도 큐브를 잡았는가 (놓침과 처음 찾기를 가른다)
let searchSince = 0; // 찾는 중이 된 시각
let poseKnown = false; // 카메라: 추적기가 큐브 방향(위·앞 색)을 한 번이라도 알려 줬는가
let levelBeforeSample: string | null = null; // 샘플 영상 때문에 '혼자 풀기'로 바꾸기 전 수준
let lastFile: File | null = null; // 다시 읽기에서 같은 영상 파일을 다시 열려고
let camPose: View | null = null; // 카메라가 마지막으로 알려 준 자세 (키보드에 다녀와도 지키려고)
let warnActive = false; // 돌리는 도중 "그 면 말고" 하고 말린 중인가

const KEYS: Record<string, [Letter, 1 | 3]> = {
  i: ["R", 1], k: ["R", 3], d: ["L", 1], e: ["L", 3],
  j: ["U", 1], f: ["U", 3], s: ["D", 1], l: ["D", 3],
  h: ["F", 1], g: ["F", 3], w: ["B", 1], o: ["B", 3],
};
const AXIS: Record<Letter, [number, number, number]> = { R: [1, 0, 0], L: [-1, 0, 0], U: [0, 1, 0], D: [0, -1, 0], F: [0, 0, 1], B: [0, 0, -1] };
const TURN_MS = 280; // 90도 도는 데 걸리는 시간 (누르고 있어야 돈다)

interface Turning {
  key: string;
  letter: Letter;
  k: 1 | 3;
  color: number;
  start: number;
  angle: number;
  released: boolean;
}
let turning: Turning | null = null;

// --- 미리 합성한 음성이 있으면 쓴다 ------------------------------------------------

const VOICE_KEY = "cubecoach.voice";
const NOTE_KEYS = "가상 큐브는 카메라 추적기와 똑같은 이벤트를 코치에 보냅니다.";
const NOTE_CAM = "카메라 추적: 직접 학습한 인식 모델(CubeSegNet)과 추적기가 이 브라우저 안에서 돌아갑니다.";
let voiceNote = "";
const note = () => `${mode === "camera" && picked ? NOTE_CAM : NOTE_KEYS} ${voiceNote}`;

/** 미리 합성한 목소리(F2 여성 / M1 남성)를 불러온다. 모든 대사를 그 목소리 하나로 읽는다(못 불러오면 전부 브라우저 음성). */
async function loadClips(name: string): Promise<void> {
  try {
    localStorage.setItem(VOICE_KEY, name);
  } catch {
    /* 저장 못 해도 이번에는 그 목소리로 */
  }
  try {
    const res = await fetch(`./tts/${name}/manifest.json`);
    if (!res.ok) throw new Error(String(res.status));
    const manifest = (await res.json()) as Record<string, string>;
    voice.setEngine(new ClipEngine(manifest, `./tts/${name}/`, web) as SpeechEngine);
    voiceNote = `음성은 미리 합성한 AI 목소리(Supertonic 3) 하나로 모든 대사를 읽어요.`;
  } catch {
    voice.setEngine(web);
    voiceNote = `음성은 브라우저 음성(${web.voiceName})이 읽어요.`;
  }
  setText($("inputNote"), note());
}

function savedVoice(): string {
  try {
    return localStorage.getItem(VOICE_KEY) ?? "F2";
  } catch {
    return "F2";
  }
}

// --- 코치와 주고받기 -------------------------------------------------------------

function emit(us: Utterance[]): void {
  // 바로잡기가 끝났으면 남은 "되돌리세요" 말을 지운다. 코치가 아무 말도 안 했을 때도, 새 말을 넣기 전에.
  voice.setFixing(!!coach?.fixing);
  if (!us.length) return;
  voice.push(us);
  for (const u of us) {
    addLog(u.text, u.kind);
    if (u.kind === "goal") lastGoal = u.text;
  }
  lastKind = us[us.length - 1].kind;
  if (us.some((u) => u.kind === "fix") && (turning || camTurn)) warnActive = true;
  // 힌트만 수준: 코치가 동작을 직접 불러 준 때부터 다음 수까지만 화면에도 보여 준다
  if (us.some((u) => u.kind === "move" || u.kind === "fix" || u.text.startsWith("처음은"))) revealAt = coach?.moves.length ?? -1;
  if (!voice.enabled) setText($("say"), us[us.length - 1].text);
  if (us.some((u) => u.kind === "summary") && coach) recordSolve(coach);
  render();
}

const recorded = new WeakSet<Coach>(); // 기록을 이미 남긴 풀이 (한 풀이는 한 번만 저장한다)

/** 다 풀었을 때: 기록을 저장하고 지난 기록과 비교해 말해 준다 */
function recordSolve(c: Coach): void {
  if (recorded.has(c)) return;
  recorded.add(c);
  const ms = c.moves;
  if (!ms.length) return;
  const rec: SolveRecord = {
    at: Date.now(),
    method: c.method,
    level: c.level,
    totalMs: ms[ms.length - 1].t - ms[0].t,
    moves: ms.length,
    mistakes: ms.filter((m) => !m.ok).length,
    splits: { ...c.splits },
    steps: [...c.steps],
  };
  saveSolve(rec);
  const history = loadHistory();
  const cmp = compareStages(rec, history);
  const fb = feedbackText(cmp);
  if (fb) {
    const u: Utterance = { text: fb, kind: "summary", priority: 7 };
    voice.push([u]);
    addLog(u.text, u.kind);
  }
  const sec = (x: number | null) => (x === null ? "–" : (x / 1000).toFixed(1));
  const rows = cmp
    .map((r) => {
      const cls = r.avg === null ? "" : r.now < r.avg ? "faster" : "slower";
      return `<tr><td>${STAGE_NAME[r.stage]}</td><td class="${cls}">${sec(r.now)}</td><td>${sec(r.avg)}</td><td>${sec(r.best)}</td></tr>`;
    })
    .join("");
  const weak = caseStats(history).slice(0, 3);
  const box = $("record");
  box.hidden = false;
  box.innerHTML =
    `<h2>이번 기록 <small>단위 초, 같은 방법으로 ${history.filter((h) => h.method === rec.method).length}번째</small></h2>` +
    `<table><tr><th>단계</th><th>이번</th><th>평균</th><th>최고</th></tr>${rows}</table>` +
    (weak.length ? `<h2>다음에 연습할 공식</h2><ul>${weak.map((w) => `<li>${w.name} <small>알아보기 ${(w.reactMs / 1000).toFixed(1)}초, 한 수에 ${(w.execMs / 1000).toFixed(2)}초</small></li>`).join("")}</ul>` : "");
}

// --- 카메라 입력 -------------------------------------------------------------------

const visionEvents = {
  status: (text: string) => {
    camErr = "";
    setText($("camStatus"), text);
    render();
  },
  scanProgress: (perSlot: number[], done: number) => {
    // 스캔 상태의 슬롯 = 가운데 색 (0 흰 1 빨 2 초 3 노 4 주 5 파)
    scanCount = perSlot.slice(0, 6);
    scanLeft = perSlot.map((n, slot) => (n >= 8 ? -1 : slot)).filter((s) => s >= 0);
    void done;
    render();
  },
  scanned: (st: State) => {
    scanned = st;
    document.body.dataset.scanned = Array.from(st).join(""); // 시험용: 읽은 54칸
    cube = st;
    scanLeft = [];
    addLog("카메라로 큐브 상태를 다 읽었어요.", "status");
    // start()가 음성 대기열을 비우므로, 알림은 코치의 첫 안내와 한 묶음으로 넘긴다
    start([{ text: "다 읽었어요. 시작할게요.", kind: "status", priority: 6 }]);
    markKnown();
  },
  turning: (color: number, angle: number) => {
    camTurn = { color, angle };
    if (coach) {
      const us = coach.onTurning(color, angle, performance.now());
      if (us.length) emit(us);
    }
    render();
  },
  turnCancel: () => {
    camTurn = null;
    warnActive = false;
    coach?.onTurnCancel();
    render();
  },
  move: (m: ColorMove, sync?: MoveSync) => {
    camTurn = null;
    // 추적기 상태도 함께 넘긴다: 놓친 회전을 앞에 끼워 넣었으면 위에 얹는 대신 그 상태로 맞춘다
    applyMove(m, performance.now(), undefined, sync?.state);
  },
  pose: (v: View) => {
    pose = v;
    camPose = v;
    poseKnown = true;
    if (coach) emit(coach.onPose(pose, performance.now()));
    if (drill) emit(drillPose(drill, pose));
    render();
  },
  // 추적 상태가 바뀔 때 (추적 담당이 넣어 준다)
  trackState: (s: string) => {
    if (s === "tracking") everTracked = true;
    if (s === "searching") searchSince = performance.now();
    render();
  },
};
const vision = new VisionInput($<HTMLCanvasElement>("camview"), visionEvents as VisionEvents);
let camTurn: { color: number; angle: number } | null = null;

/** 추적 담당이 아직 안 넣었을 수도 있는 것들 */
interface VisionExtra {
  setHint?(m: ColorMove | null): void;
  trackState?: string;
  quality?: number;
}
const vx = (): VisionExtra => vision as unknown as VisionExtra;

let lastHint = "\u0000";
/** 카메라 화면 위 진짜 큐브에 "다음에 돌릴 층"을 표시해 달라고 알려 준다 */
function syncHint(): void {
  const fn = vx().setHint;
  if (typeof fn !== "function") return;
  const m = mode === "camera" && showMove() ? expectedMove() : null;
  const key = m ? `${m.color}/${m.k}` : "";
  if (key === lastHint) return;
  lastHint = key;
  fn.call(vision, m);
}

/** 풀이를 비운다 (입력을 바꾸거나 처음으로 돌아갈 때) */
function resetSession(): void {
  voice.cancelAll();
  coach = null;
  drill = null;
  turning = null;
  camTurn = null;
  lastKind = "";
  lastGoal = "";
  trail = [];
  $("record").hidden = true;
  setText($("say"), "");
}

function setMode(m: InputMode): void {
  const changed = !picked || m !== mode;
  mode = m;
  picked = true;
  document.body.dataset.mode = m;
  $("modeCam").setAttribute("aria-checked", String(m === "camera"));
  $("modeKey").setAttribute("aria-checked", String(m === "keyboard"));
  if (changed) {
    resetSession();
    if (m === "keyboard") {
      vision.stop();
      camSource = null;
      scanned = null;
      scanLeft = [];
      cube = SOLVED;
      pose = { up: 3, front: 2 };
      if (levelBeforeSample) {
        // 샘플 영상 때문에 '혼자 풀기'로 바꿨던 수준을 되돌린다
        $<HTMLSelectElement>("level").value = levelBeforeSample;
        levelBeforeSample = null;
        levelNote();
      }
    } else {
      cube = SOLVED;
      pose = camPose ?? { up: 3, front: 2 };
      poseKnown = !!camPose;
    }
  }
  camErr = "";
  setText($("inputNote"), note());
  setSetupOpen(false); // 고르면 바로 다음 단계로
  syncHint();
  render();
  // 다음에 누를 곳으로 초점을 옮긴다 (키보드·화면 읽기 사용자)
  requestAnimationFrame(() => (m === "camera" ? (scanned ? null : $("camStart")) : $("scramble"))?.focus({ preventScroll: true }));
}

/** 브라우저가 주는 오류를 쉬운 말로 */
function errWhy(err: Error): string {
  switch (err.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "카메라 권한이 막혀 있어요. 주소창 왼쪽의 카메라 아이콘에서 '허용'으로 바꾼 뒤 다시 눌러 주세요.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "연결된 카메라를 찾지 못했어요.";
    case "NotReadableError":
    case "AbortError":
      return "다른 앱이 카메라를 쓰고 있어요. 그 앱을 닫고 다시 눌러 주세요.";
    case "NotSupportedError":
      return "이 브라우저에서 열 수 없는 영상 형식이에요.";
    default:
      return `(${err.message})`;
  }
}

function openSource(kind: "camera" | "sample" | "file", run: () => Promise<void>, fail: string): void {
  picked = true;
  camSource = kind;
  everTracked = false;
  if (kind !== "sample" && levelBeforeSample) {
    // 샘플 때문에 바꿨던 수준을 되돌린다 (진짜 카메라·영상은 코치 지시를 따를 수 있다)
    $<HTMLSelectElement>("level").value = levelBeforeSample;
    levelBeforeSample = null;
    levelNote();
  }
  scanned = null;
  scanLeft = [];
  scanCount = [0, 0, 0, 0, 0, 0];
  resetSession();
  setSetupOpen(false);
  run().catch((err: Error) => {
    camSource = null;
    camErr = `${fail}. ${errWhy(err)}${kind === "camera" ? " 카메라가 없으면 '샘플 영상'으로 볼 수 있어요." : ""}`;
    setText($("camStatus"), camErr);
    render();
  });
  render();
}

$("modeCam").addEventListener("click", () => setMode("camera"));
$("modeKey").addEventListener("click", () => setMode("keyboard"));
$("camStart").addEventListener("click", () => openSource("camera", () => vision.startCamera(), "카메라를 열 수 없어요"));
const openFile = (f: File) => openSource("file", () => vision.startFile(f), "영상을 열 수 없어요");
const openSample = () => openSource("sample", () => vision.startUrl("./samples/sexy_move_46s.mp4"), "샘플 영상을 열 수 없어요");
$<HTMLInputElement>("camFile").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) {
    lastFile = f;
    openFile(f);
  }
});
$("camSample").addEventListener("click", () => {
  // 샘플 영상 속 사람은 코치 지시가 아니라 정해진 동작(섹시무브 등)을 하므로, 지시 대신 관찰 모드로 보여 준다
  const lv = $<HTMLSelectElement>("level");
  if (lv.value !== "observe") levelBeforeSample = lv.value;
  lv.value = "observe";
  levelNote();
  openSample();
  addLog("샘플 영상은 코치 지시를 따르지 않는 영상이라 '혼자 풀기'로 보여 줘요. 돌린 면이 하나씩 찍힙니다.", "status");
});
$("camRescan").addEventListener("click", () => {
  // 영상이 끝나 멈춰 있으면 같은 영상을 처음부터 다시 연다
  if (trackNow() === "idle" && camSource) {
    if (camSource === "sample") return openSample();
    if (camSource === "file" && lastFile) return openFile(lastFile);
    if (camSource === "camera") return openSource("camera", () => vision.startCamera(), "카메라를 열 수 없어요");
  }
  scanned = null;
  scanLeft = [];
  scanCount = [0, 0, 0, 0, 0, 0];
  everTracked = false;
  resetSession();
  vision.rescan();
  render();
});
setInterval(() => {
  if (mode === "camera" && vision.phase !== "idle") setText($("camFps"), `${vision.fps} fps`);
  if (mode === "camera") renderTrack(); // 찾는 중이 길어지면 안내를 띄운다
}, 500);

function start(prelude: Utterance[] = []): void {
  const method = $<HTMLSelectElement>("method").value as Method;
  const level = $<HTMLSelectElement>("level").value as Level | "drill";
  pose = mode === "camera" ? pose : { up: 3, front: 2 };
  voice.cancelAll();
  $("log").innerHTML = "";
  $("record").hidden = true;
  coach = null;
  drill = null;
  lastKind = "";
  lastGoal = "";
  revealAt = -1;
  trail = [];
  if (mode === "camera" && !scanned) {
    setText($("camStatus"), "먼저 카메라를 켜고 큐브 여섯 면을 읽어 주세요.");
    render();
    return;
  }
  setSetupOpen(false); // 풀이 중에는 설정을 접는다
  if (level === "drill") {
    // 공식 연습: 다 맞춘 큐브에서, 기록상 약한 공식부터 (카메라면 지금 큐브에서)
    cube = mode === "camera" ? scanned! : SOLVED;
    const kind = method === "cfop" ? "pll" : "oll";
    const pool = kind === "pll" ? PLL : OCLL;
    const weak = caseStats(loadHistory()).map((w) => w.name);
    const rank = (name: string) => {
      const i = weak.findIndex((w) => w.endsWith(" " + name));
      return i < 0 ? weak.length : i;
    };
    const cases = [...pool].sort((a, b) => rank(a.name) - rank(b.name));
    drill = new Drill(cube, cases, kind);
    drill.pose = pose;
    emit([...prelude, ...drill.start(performance.now())]);
    return;
  }
  cube = mode === "camera" ? scanned! : applySlotAlg(SOLVED, randomScramble(25));
  coach = new Coach(cube, { method, level });
  coach.pose = pose;
  emit([...prelude, ...coach.start(performance.now())]);
  render();
}

function commit(t: Turning, now: number): void {
  applyMove({ color: t.color, k: t.k }, now, `${t.letter}${t.k === 3 ? "'" : ""}`);
}

/**
 * 회전 하나가 끝났다 (키보드 또는 카메라 추적기).
 * tracked: 카메라 추적기 상태 (키보드는 없음). 추적기가 놓친 회전을 몇 수 앞에 끼워 넣었으면 위에 얹은 결과와 달라진다.
 * 그때는 화면·코치·연습을 추적기 상태로 맞춘다 (coach/resync.ts). 키보드는 예전과 똑같이 위에 얹는다.
 */
function applyMove(m: ColorMove, now: number, label?: string, tracked?: State): void {
  warnActive = false;
  // 기호는 돌리기 전 방향으로 읽는다 (돌린 뒤에는 그 면의 위치가 같아도 상태가 바뀐다)
  const sym = label ?? moveSymbol(cube, curPose(), m);
  const next = followMove(cube, m, tracked);
  cube = next.state;
  if (mode === "camera") scanned = cube;
  addLog(sym, "user");
  trail.push(sym);
  if (trail.length > 12) trail.shift();
  if (coach) emit(next.resync ? coach.resync(cube, m, now) : coach.onMove(m, now));
  if (drill) {
    emit(next.resync ? drill.resync(cube, m, now) : drill.onMove(m, now));
    renderDrill();
  }
  markKnown();
  render();
}

/** 시험용(scripts/browser_sample_check.sh): 화면이 아는 큐브와 코치·연습이 아는 큐브 (54칸, 카메라 모드만) */
function markKnown(): void {
  if (mode !== "camera") return;
  document.body.dataset.cube = Array.from(cube).join("");
  const s = coach?.s ?? drill?.s;
  document.body.dataset.coach = s ? Array.from(s).join("") : "";
}

function rotateCube(axis: string, k: number): void {
  pose = rotateView(cube, pose, axis, k);
  if (coach) emit(coach.onPose(pose, performance.now()));
  if (drill) emit(drillPose(drill, pose));
  render();
}

/** 공식 연습에 쥔 방향을 알린다 (코칭 담당의 Drill.onPose가 있으면 그것으로) */
function drillPose(d: Drill, v: View): Utterance[] {
  const fn = (d as unknown as { onPose?: (v: View, t: number) => Utterance[] }).onPose;
  if (typeof fn === "function") return fn.call(d, v, performance.now());
  d.pose = v;
  return [];
}

// --- 입력 -----------------------------------------------------------------------

function press(key: string): void {
  if (turning || !KEYS[key]) return;
  const [letter, k] = KEYS[key];
  turning = { key, letter, k, color: viewColors(cube, pose)[letter], start: performance.now(), angle: 0, released: false };
  requestAnimationFrame(step);
}

function release(key: string): void {
  if (!turning || turning.key !== key) return;
  turning.released = true;
  if (Math.abs(turning.angle) < Math.PI / 4) {
    // 반도 안 돌리고 놓았다: 취소
    turning = null;
    warnActive = false;
    coach?.onTurnCancel();
    render();
  }
}

function step(now: number): void {
  const t = turning;
  if (!t) return;
  const frac = Math.min(1, (now - t.start) / TURN_MS);
  t.angle = (t.k === 1 ? 1 : -1) * frac * (Math.PI / 2);
  if (coach) {
    const us = coach.onTurning(t.color, t.angle, now);
    if (us.length) emit(us);
  }
  if (frac >= 1) {
    turning = null;
    commit(t, now);
    return;
  }
  render();
  requestAnimationFrame(step);
}

/** 설정·선택 칸에 손이 가 있을 때는 큐브를 돌리지 않는다 */
const inForm = (el: EventTarget | null) =>
  el instanceof HTMLElement && (el instanceof HTMLSelectElement || el instanceof HTMLInputElement || !!el.closest(".setup"));

window.addEventListener("keydown", (e) => {
  if (inForm(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (!picked || mode !== "keyboard") return; // 키보드는 가상 큐브 모드에서만
  const k = e.key.toLowerCase();
  if (KEYS[k]) {
    e.preventDefault();
    if (!e.repeat) press(k);
  } else if (e.key.startsWith("Arrow")) {
    e.preventDefault();
    if (e.key === "ArrowLeft") rotateCube("y", 1);
    else if (e.key === "ArrowRight") rotateCube("y", -1);
    else if (e.key === "ArrowUp") rotateCube("x", 1);
    else if (e.key === "ArrowDown") rotateCube("x", -1);
  }
});
window.addEventListener("keyup", (e) => release(e.key.toLowerCase()));

// --- 대화 기록 -------------------------------------------------------------------

let logN = 0;
function addLog(text: string, kind: string): void {
  const li = document.createElement("li");
  li.className = kind;
  li.dataset.text = text;
  const label: Record<string, string> = { move: "지시", fix: "바로잡기", hold: "방향", goal: "목표", intro: "설명", stage: "단계", praise: "칭찬", hint: "힌트", status: "상태", summary: "정리" };
  if (kind === "user") li.textContent = `돌림 ${text}`;
  else {
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = label[kind] ?? kind;
    li.append(k, document.createTextNode(text));
  }
  const log = $("log");
  log.appendChild(li);
  log.scrollTop = log.scrollHeight;
  logN = log.childElementCount;
  setText($("logCount"), `${logN}줄`);
}

voice.onSpeak = (u) => {
  document.querySelectorAll(".log li.speaking").forEach((el) => el.classList.remove("speaking"));
  const items = [...document.querySelectorAll<HTMLLIElement>(".log li")].reverse();
  items.find((el) => el.dataset.text === u.text)?.classList.add("speaking");
  setText($("say"), u.text); // 음성과 같은 내용을 화면에도
};

// --- 지금 어느 단계인가 -------------------------------------------------------------

type StepId = "pick" | "camera" | "scan" | "start" | "hold" | "solve" | "done";
const STEP_LIST: Record<InputMode, { id: StepId; name: string; desc: string }[]> = {
  camera: [
    { id: "pick", name: "입력 고르기", desc: "카메라 · 영상을 골랐어요." },
    { id: "camera", name: "카메라 켜기", desc: "권한을 허용하면 내 모습이 거울처럼 나와요. 큐브가 없으면 샘플 영상." },
    { id: "scan", name: "여섯 면 읽기", desc: "돌리지 말고 한 면씩 카메라 정면에 1초씩. 순서와 방향은 상관없어요." },
    { id: "hold", name: "잡는 법", desc: "위·앞이 무슨 색인지 확인하고 그 방향 그대로 잡아요. 안내는 늘 이 방향 기준이에요." },
    { id: "solve", name: "풀기", desc: "카드의 기호대로 돌려요. 틀리면 빨간 카드로 바로잡아 줘요." },
    { id: "done", name: "기록", desc: "단계별 시간을 지난 기록과 비교해요." },
  ],
  keyboard: [
    { id: "pick", name: "입력 고르기", desc: "가상 큐브를 골랐어요." },
    { id: "start", name: "섞고 시작", desc: "무작위로 25번 섞은 큐브로 시작해요." },
    { id: "hold", name: "잡는 법", desc: "위·앞이 무슨 색인지 확인해요. 화살표 키로 통째로 돌릴 수 있어요." },
    { id: "solve", name: "풀기", desc: "글자 키를 누르고 있으면 그 면이 돌아가요." },
    { id: "done", name: "기록", desc: "단계별 시간을 지난 기록과 비교해요." },
  ],
};

const curPose = (): View => coach?.pose ?? drill?.pose ?? pose;

/** 코치가 잡아 달라고 하고 기다리는 방향 (코칭 담당의 coach.grip, 없으면 첫 수 전의 단계 방향) */
function gripView(): View | null {
  if (!coach || coach.level === "observe") return null;
  const c = coach as unknown as { grip?: View | null };
  if ("grip" in coach || c.grip !== undefined) return c.grip ?? null;
  return holdPending() ? coach.currentStep!.view! : null;
}

/** 코치가 "이렇게 잡으세요"라고 한 방향으로 아직 안 잡았는가 (첫 수 전, 예전 방식) */
function holdPending(): boolean {
  if (!coach || coach.level === "observe") return false;
  if (coach.moves.length > 0) return false;
  const v = coach.currentStep?.view;
  if (!v) return false;
  const vc = viewColors(cube, curPose());
  return !(vc.U === v.up && vc.F === v.front);
}

/** 첫 수를 두기 전: 기준 면(위·앞 색)을 확인하고 그 방향으로 잡는 단계 */
function firstMove(): boolean {
  return !!coach && coach.level !== "observe" && coach.moves.length === 0 && coach.currentStage !== "done";
}

function uiStep(): StepId {
  if (coach?.currentStage === "done") return "done";
  if (coach || drill) return firstMove() ? "hold" : "solve";
  if (!picked) return "pick";
  if (mode === "keyboard") return "start";
  const ph = vision.phase;
  if (ph === "idle" || ph === "loading") return "camera";
  return "scan"; // 면 읽는 중, 또는 다 읽고 시작을 기다리는 중
}

/** 화면에 동작을 보여 줄 때인가 (힌트만 수준은 코치가 불러 준 뒤에만) */
function showMove(): boolean {
  if (drill) return !!drill.expected();
  if (!coach || !coach.expected()) return false;
  if (coach.level === "observe") return false;
  if (coach.level === "hint") return revealAt === coach.moves.length;
  return true;
}

/** 잡는 방향을 바꿔 달라는 한 문장 */
function gripSentence(v: View): string {
  return `먼저 ${iGa(COLOR_KO[v.up])} 위, ${iGa(COLOR_KO[v.front])} 앞에 오게 ${mode === "camera" ? "손에 든 큐브를 돌려 잡으세요." : "화살표 키로 돌려 잡으세요."}`;
}

/**
 * 지금 덩어리의 수들을 사람이 든 방향 기준 기호로 (음성과 같게).
 * 한 수씩 부를 때 음성은 두 줄 돌리기(r)를 같은 효과의 한 층(L)으로 부르므로, 화면도 한 층 기호로 맞춘다.
 * 지금 둘 수는 굵게.
 */
function stepSymbols(limit = Infinity): string {
  const st = coach?.currentStep;
  if (!coach || !st || !st.moves.length) return "";
  const pm = (coach as unknown as { pendingMove?: () => unknown }).pendingMove;
  const cur = typeof pm === "function" ? pm.call(coach) : null;
  const at = cur ? st.moves.indexOf(cur as (typeof st.moves)[number]) : -1;
  // limit: 앞에서부터 이만큼만 보여 주고 나머지는 … (힌트만 수준은 코치가 불러 준 수까지만)
  const syms = st.moves.slice(0, limit).map((m, i) => {
    const sym = moveSymbol(cube, curPose(), { color: m.color, k: m.k });
    return i === at ? `<b>${sym}</b>` : i < at ? `<s>${sym}</s>` : sym;
  });
  return syms.join(" ") + (limit < st.moves.length ? " …" : "");
}

/** 지금 무엇을 하면 되는지 한 문장 */
function todoText(st: StepId): string {
  if (camErr) return st === "camera" ? "카메라나 영상을 열지 못했어요. 아래 안내를 보고 다시 해 주세요." : camErr;
  if (mode === "camera" && vision.phase !== "idle" && vision.phase !== "loading" && trackNow() === "idle")
    return "영상이 끝났어요. '다시 읽기'나 '샘플 영상'으로 처음부터 다시 볼 수 있어요.";
  switch (st) {
    case "pick":
      return "무엇으로 할지 고르세요. 큐브가 있으면 '카메라 · 영상', 없으면 '가상 큐브'예요.";
    case "camera":
      return vision.phase === "loading"
        ? "인식 모델을 불러오는 중이에요. 잠깐만요."
        : "'카메라 켜기'를 누르고 권한을 허용해 주세요. 카메라나 큐브가 없으면 '샘플 영상'으로 볼 수 있어요.";
    case "scan":
      if (scanned) return "다 읽었어요. 곧 안내를 시작해요.";
      return scanLeft.length && scanLeft.length < 6
        ? `돌리지 말고 ${scanLeft.map((c) => COLOR_KO[c]).join(", ")} 면을 카메라 정면에 1초씩 보여 주세요.`
        : "돌리지 말고 여섯 면을 한 면씩 카메라 정면에 1초씩 보여 주세요. 순서와 방향은 상관없어요.";
    case "start":
      return "준비되면 '섞고 시작'을 누르세요. 코치가 첫 동작부터 소리와 카드로 알려 줘요.";
    case "hold": {
      if (!gripView() || isFixing()) {
        if (!poseReady()) return "큐브를 카메라 앞에 그대로 들고 있어 주세요. 위·앞이 무슨 색인지 읽고 있어요.";
        const r = reference();
        return `${iGa(COLOR_KO[r.up])} 위, ${iGa(COLOR_KO[r.front])} 앞인 지금 방향 그대로 잡고 시작해요. 안내는 늘 내가 보는 이 방향 기준이에요.`;
      }
      return gripSentence(gripView()!);
    }
    case "solve": {
      if (drill)
        return drill.phase === "setup"
          ? "공식 연습: 부르는 대로 돌려서 연습할 상황을 만들어요."
          : drill.phase === "restore"
            ? "공식 연습: 먼저 1·2층까지 맞춰요."
            : "공식 연습: 어떤 상황인지 알아보고 공식으로 풀어 보세요. 시간을 재고 있어요.";
      if (coach?.level === "observe")
        return mode === "camera" ? "혼자 풀기: 지시 없이, 돌리는 대로 따라가며 단계별 시간을 재요." : "혼자 풀기: 지시 없이 단계별 시간을 재요.";
      const g = gripView();
      if (g && !isFixing()) return gripSentence(g);
      if (lastGoal) return `${STAGE_NAME[coach!.currentStage]}: ${lastGoal}`;
      const m = expectedMove();
      return m ? moveText(cube, curPose(), m, "full") : "잠시만요.";
    }
    case "done":
      return "다 맞췄어요! 기록 칸에서 단계별 시간을 지난 기록과 비교해 보세요.";
  }
}

/** 입력을 고르기 전에는 어느 쪽에도 맞는 큰 흐름만 */
const STEP_ANY: { id: StepId; name: string; desc: string }[] = [
  { id: "pick", name: "입력 고르기", desc: "" },
  { id: "start", name: "준비", desc: "" },
  { id: "hold", name: "잡는 법", desc: "" },
  { id: "solve", name: "풀기", desc: "" },
  { id: "done", name: "기록", desc: "" },
];

function renderStepper(st: StepId): void {
  const list = picked ? STEP_LIST[mode] : STEP_ANY;
  const idx = Math.max(0, list.findIndex((s) => s.id === st));
  setHtml(
    $("stepper"),
    `<ol>${list
      .map((s, i) => {
        const cls = i < idx ? "is-done" : i === idx ? "is-now" : "is-next";
        const state = i < idx ? "끝남" : i === idx ? "지금" : "다음";
        return `<li class="st ${cls}"${i === idx ? ' aria-current="step"' : ""}><i aria-hidden="true">${i < idx ? "✓" : i + 1}</i><span class="nm">${s.name}</span><span class="sr"> (${state})</span></li>`;
      })
      .join("")}</ol>`,
  );
  // 시작 전에는 옆 칸에 진행 순서를 자세히
  setHtml(
    $("flowList"),
    list
      .map((s, i) => `<li class="${i < idx ? "is-done" : i === idx ? "is-now" : ""}"><b>${s.name}</b><span>${s.desc}</span></li>`)
      .join(""),
  );
}

// --- 추적 상태 --------------------------------------------------------------------

const PHASE_TO_TRACK: Record<string, string> = { idle: "idle", loading: "loading", scan: "scanning", track: "tracking" };

function trackNow(): string {
  const s = vx().trackState;
  return typeof s === "string" ? s : PHASE_TO_TRACK[vision.phase] ?? "idle";
}

function renderTrack(): void {
  const lost = $("camLost");
  const on = mode === "camera" && vision.phase !== "idle" && vision.phase !== "loading";
  show($("track"), on);
  show($("camFps"), on);
  if (!on) {
    show(lost, false);
    return;
  }
  const s = trackNow();
  const ended = s === "idle"; // 영상이 끝났다 (화면은 마지막 장면에 멈춰 있다)
  const missing = s === "searching" && everTracked;
  const longSearch = s === "searching" && performance.now() - searchSince > 1200;
  const shown = ended ? "ended" : missing ? "lost" : s;
  const TXT: Record<string, string> = {
    loading: "모델 불러오는 중",
    scanning: "면 읽는 중",
    searching: "큐브 찾는 중",
    lost: "놓침",
    tracking: "내 큐브 따라가는 중",
    ended: "영상 끝",
  };
  const el = $("track");
  setData(el, "s", s); // 원래 상태 (다른 스크립트가 읽는다)
  setData(el, "show", shown); // 화면에 보여 주는 상태 (놓침·영상 끝 포함)
  setText($("trackText"), TXT[shown] ?? s);
  const q = vx().quality;
  const bar = $("qbar");
  show(bar, typeof q === "number" && s === "tracking");
  if (typeof q === "number") {
    const w = `${Math.round(Math.max(0, Math.min(1, q)) * 100)}%`;
    const i = bar.firstElementChild as HTMLElement | null;
    if (i && i.style.width !== w) i.style.width = w;
    bar.title = `맞는 정도 ${w}`;
  }
  // 화면 가운데 안내: 놓쳤거나 오래 못 찾을 때, 영상이 끝났을 때
  const msg = ended
    ? camSource === "camera"
      ? "카메라가 멈췄어요. '카메라 켜기'를 다시 눌러 주세요."
      : "영상이 끝났어요. '샘플 영상'을 누르면 처음부터 다시 봐요."
    : missing || longSearch
      ? "큐브를 놓쳤어요. 큐브를 화면 가운데로, 조금 가까이 보여 주세요."
      : "";
  show(lost, !!msg);
  if (msg) setText(lost, msg);
  setData(lost, "kind", ended ? "ended" : "lost");
  // 잘 따라가는 동안에는 지난 안내 글이 남아 헷갈리지 않게 숨긴다
  // 한 문장 안내(todo)와 같은 말이면 두 번 보이지 않게 숨긴다
  const st = $("camStatus").textContent ?? "";
  const same = /^(돌리지 말고|인식 모델을 불러오는|영상이 끝났어요)/.test(st);
  show($("camStatus"), !!camErr || (s !== "tracking" && !msg && !!st && !same));
}

/** 카메라 켜기 칸 안의 오류 안내 */
function renderCamErr(): void {
  const el = $("camErrMsg");
  setText(el, camErr);
  show(el, !!camErr);
}

// --- 다음 할 일 카드 ----------------------------------------------------------------

function expectedMove(): ColorMove | null {
  return coach?.expected() ?? drill?.expected() ?? null;
}

/** 기준 면 (사람 시점의 위·앞·오른쪽 가운데 색). 코치가 알려 주면 그것을, 없으면 지금 방향에서 계산한다. */
function reference(): { up: number; front: number; right: number } {
  const c = coach as unknown as { reference?: () => { up: number; front: number; right: number } } | null;
  if (c && typeof c.reference === "function") {
    try {
      const r = c.reference();
      if (r) return r;
    } catch {
      /* 코치가 아직 못 정했으면 아래로 */
    }
  }
  const vc = viewColors(cube, curPose());
  return { up: vc.U, front: vc.F, right: vc.R };
}

/** 기준 면을 보여 줄 수 있는가 (카메라는 추적기가 방향을 알려 준 뒤에만) */
const poseReady = () => mode !== "camera" || poseKnown;

/**
 * 코치가 지금 바로잡기를 시키는 중인가: 되돌릴 수가 남았거나, 돌리는 도중 "그 면 말고" 하고 말린 중.
 * 되돌릴 수: expected()는 있는데 공식의 다음 수(pendingMove)가 없으면 되돌리기다.
 */
function isFixing(): boolean {
  const c = coach as unknown as { fixing?: boolean; pendingMove?: () => unknown } | null;
  if (!coach || !c) return false;
  if (typeof c.fixing === "boolean") return c.fixing || warnActive;
  if (typeof c.pendingMove === "function") return (!!coach.expected() && !c.pendingMove.call(coach)) || warnActive;
  return lastKind === "fix";
}

const sw = (c: number) => `<i class="sw" style="background:${HEX[c]}"></i>`;
const chip = (label: string, c: number) => `<span class="chip">${sw(c)}<span>${label ? `<small>${label}</small> ` : ""}${COLOR_KO[c]}</span></span>`;

interface Card {
  state: string;
  badge: string;
  fig: string;
  sym: string;
  text: string;
  face: string;
  refs: string;
}

function paintNext(st: StepId): void {
  const exp = expectedMove();
  const card: Card = { state: "wait", badge: "", fig: "", sym: "", text: "", face: "", refs: "" };
  const refsNow = () => {
    if (!poseReady()) return `<b>내 큐브 기준</b><span class="refs-wait">위·앞 색을 읽는 중이에요</span>`;
    const r = reference();
    return `<b>내 큐브 기준</b>${chip("위", r.up)}${chip("앞", r.front)}${chip("오른쪽", r.right)}`;
  };
  const moveCard = (m: ColorMove) => {
    const { letter, wide } = moveLetter(cube, curPose(), m);
    const r = reference();
    card.sym = moveSymbol(cube, curPose(), m);
    card.text = moveText(cube, curPose(), m, "full");
    card.fig = renderMoveArrow(AXIS[letter], m.k, wide, 104, poseReady() ? r : undefined);
    card.face = wide ? "" : `${sw(m.color)}<span>돌릴 면: 가운데가 <b>${COLOR_KO[m.color]}</b>인 면</span>`;
    card.refs = refsNow();
  };

  if (st === "done") {
    card.state = "done";
    card.badge = "완성";
    card.sym = "✓";
    const ms = coach?.moves ?? [];
    const sec = ms.length > 1 ? (ms[ms.length - 1].t - ms[0].t) / 1000 : 0;
    card.text = ms.length ? `다 맞췄어요! ${ms.length}수, ${sec >= 60 ? `${Math.floor(sec / 60)}분 ${Math.round(sec % 60)}초` : `${sec.toFixed(1)}초`}` : "다 맞췄어요!";
  } else if (st === "scan") {
    card.state = "scan";
    card.badge = scanned ? "다 읽었어요" : "여섯 면 읽기 · 돌리지 마세요";
    card.sym = `${6 - scanLeft.length}/6`;
    const left = scanLeft.length ? scanLeft : [0, 1, 2, 3, 4, 5];
    card.text = scanned
      ? "다 읽었어요. 곧 안내를 시작해요."
      : `${left.map((c) => COLOR_KO[c]).join(", ")} 면을 하나씩 카메라 정면에 1초씩 보여 주세요.`;
  } else if (gripView() && !isFixing()) {
    // 되돌리기가 먼저: 바로잡는 중이면 잡는 법보다 바로잡기 카드를 보여 준다
    const v = gripView()!;
    card.state = "hold";
    card.badge = "먼저 이렇게 잡기";
    card.text = `${iGa(COLOR_KO[v.up])} 위, ${iGa(COLOR_KO[v.front])} 앞에 오게 잡으세요`;
    card.fig = `<div class="hold-pic">${renderCube(cube, v, null, 120)}</div>`;
    const now = viewColors(cube, curPose());
    card.face = poseReady()
      ? `<span>지금은 ${iGa(COLOR_KO[now.U])} 위, ${iGa(COLOR_KO[now.F])} 앞이에요.${mode === "keyboard" ? " 화살표 키로 돌려요." : ""}</span>`
      : "<span>지금 든 방향을 읽는 중이에요.</span>";
    card.refs = `<b>이렇게</b>${chip("위", v.up)}${chip("앞", v.front)}`;
  } else if (drill) {
    card.state = "move";
    card.badge = drill.phase === "setup" ? "공식 연습 · 준비 동작" : drill.phase === "restore" ? "공식 연습 · 되돌리는 중" : "공식 연습 · 시간 재는 중";
    if (exp) moveCard(exp);
    else {
      // 알아보기 연습이므로 풀기 전에는 공식 이름을 보여 주지 않는다
      card.state = "observe";
      card.sym = "?";
      card.text = drill.phase === "solve" ? "어떤 상황인지 알아보고 공식으로 풀어 보세요" : "잠시만요";
      card.refs = refsNow();
    }
  } else if (coach && coach.level === "observe") {
    card.state = "observe";
    card.badge = "혼자 풀기 · 시간 재는 중";
    card.sym = trail.length ? trail[trail.length - 1] : "";
    card.text = trail.length ? (mode === "camera" ? "방금 인식한 회전이에요" : "방금 돌린 회전이에요") : mode === "camera" ? "돌리는 대로 따라가요" : "혼자 풀어 보세요";
    card.refs = refsNow();
  } else if (coach && exp && showMove()) {
    const fixing = isFixing();
    card.state = fixing ? "fix" : "move";
    card.badge = fixing ? "바로잡기 · 이것부터 해 주세요" : st === "hold" ? "첫 동작 · 위·앞 색을 확인하고 시작해요" : "다음 동작";
    moveCard(exp);
  } else if (coach && exp) {
    card.state = "observe";
    card.badge = "힌트만 · 혼자 해 보세요";
    card.text = "막히면 코치가 힌트를 줘요";
    card.refs = refsNow();
  } else if (st === "start") {
    card.state = "wait";
    card.badge = "준비";
    card.text = "섞은 큐브로 시작해요";
    card.face = `<span>${levelSummary()}</span>`;
  }

  const next = $("next");
  setData(next, "state", card.state);
  setData(next, "first", st === "hold" && !gripView() ? "1" : "");
  setText($("nextBadge"), card.badge);
  setHtml($("nextArrow"), card.fig);
  setText($("nextSym"), card.sym);
  setText($("nextText"), card.text);
  setHtml($("nextFace"), card.face);
  show($("nextFace"), !!card.face);
  setHtml($("refs"), card.refs);
  show($("refs"), !!card.refs);

  // 면 읽기 진행: 여섯 면 칸
  const scanning = st === "scan" && !scanned;
  show($("scanFaces"), scanning);
  if (scanning)
    setHtml(
      $("scanFaces"),
      scanCount
        .map((n, slot) => `<span class="sf ${n >= 8 ? "ok" : ""}" style="--c:${HEX[slot]}"><b><i style="height:${(100 * n) / 8}%"></i></b><small>${COLOR_KO[slot]}</small></span>`)
        .join(""),
    );

  // 혼자 풀기에서는 인식한 회전을 줄로 보여 준다
  const showTrail = card.state === "observe" && trail.length > 0 && !drill;
  show($("trail"), showTrail);
  if (showTrail) setHtml($("trail"), trail.map((s, i) => `<li${i === trail.length - 1 ? ' class="last"' : ""}>${s}</li>`).join(" "));

  // 시작 버튼: 시작 전·완성 뒤에만 크게, 풀이 중에는 숨긴다 (실수로 풀이를 지우지 않게)
  const sc = $("scramble");
  const showStart = st === "start" || st === "done";
  show(sc, showStart);
  setText(sc, st === "done" ? (mode === "camera" ? "지금 상태로 다시 시작" : "한 번 더 섞기") : "섞고 시작");
}

// --- 화면 -----------------------------------------------------------------------

function renderDrill(): void {
  if (!drill) return;
  const rs = drill.results;
  if (!rs.length) return;
  const box = $("record");
  box.hidden = false;
  box.innerHTML =
    `<h2>공식 연습 <small>${rs.length}번째, 단위 초</small></h2><table><tr><th>공식</th><th>알아보기</th><th>실행</th><th>한 번에</th></tr>` +
    rs
      .slice(-6)
      .reverse()
      .map((r) => `<tr><td>${r.name}</td><td>${(r.reactMs / 1000).toFixed(1)}</td><td>${(r.execMs / 1000).toFixed(1)}</td><td>${r.clean ? "○" : "×"}</td></tr>`)
      .join("") +
    "</table>";
}

function stageNote(): string {
  if (mode === "keyboard") return "보이는 그대로가 내가 든 방향이에요. 왼쪽 아래 면이 앞(F), 오른쪽 아래 면이 오른쪽(R).";
  if (camSource === "camera") return "안내는 손에 든 큐브를 내가 보는 방향 기준이에요. 화면은 거울처럼 보여서, 화면 속 오른쪽이 내 오른손 쪽이에요.";
  if (camSource) return "안내는 큐브를 든 사람이 보는 방향 기준이에요. 영상은 카메라 쪽에서 본 모습이에요.";
  return "안내는 화면이 아니라 손에 든 큐브를 내가 보는 방향 기준이에요.";
}

function render(): void {
  const st = uiStep();
  setData(document.body, "step", st);
  renderStepper(st);
  setText($("todo"), todoText(st));
  renderTrack();
  renderCamErr();
  paintNext(st);
  syncHint();
  setText($("stageTitle"), mode === "camera" ? "내 큐브 (카메라 화면)" : "가상 큐브");
  setText($(mode === "camera" ? "camNote" : "stageNote"), stageNote());
  show($("restart"), !!(coach || drill) || (mode === "keyboard" && picked));
  show($("flow"), picked && !coach && !drill);

  // 큐브 그림: 키보드 모드에서는 주 화면, 카메라 모드에서는 접어 둔 확인용
  const exp = expectedMove();
  let hl = null as Parameters<typeof renderCube>[2];
  if (turning) hl = { axis: AXIS[turning.letter], angle: turning.angle, mode: "turning" };
  else if (camTurn) {
    const letter = (Object.entries(viewColors(cube, curPose())).find(([, c]) => c === camTurn!.color)?.[0] ?? "U") as Letter;
    hl = { axis: AXIS[letter], angle: camTurn.angle, mode: "turning" };
  } else if (exp && showMove()) {
    hl = { axis: AXIS[moveLetter(cube, curPose(), exp).letter], mode: $("next").dataset.state === "fix" ? "fix" : "next" };
  }
  const known = $<HTMLDetailsElement>("known");
  if (mode === "keyboard") {
    setHtml($("cube"), renderCube(cube, curPose(), hl));
    setHtml($("hidden"), renderHidden(cube, curPose()));
  } else if (known.open) {
    setHtml($("knownCube"), renderCube(cube, curPose(), hl, 220));
    setHtml($("knownHidden"), renderHidden(cube, curPose()));
  }
  setText(
    $("turning"),
    turning
      ? `${turning.letter}${turning.k === 3 ? "'" : ""} 돌리는 중 ${Math.round(Math.abs(turning.angle) * (180 / Math.PI))}°`
      : camTurn
        ? `${COLOR_KO[camTurn.color]} 면 돌리는 중 ${Math.round(camTurn.angle * (180 / Math.PI))}°`
        : "",
  );

  // 단계 진행 막대 (풀이를 시작한 뒤에만 뜻이 있다)
  show($("plan"), !!(coach || drill));
  if (!coach && !drill) return;
  const method = (coach?.method ?? $<HTMLSelectElement>("method").value) as Method;
  const a = analyze(cube);
  const cur = stageOf(a, method);
  const order = STAGES[method];
  const curIdx = cur === "done" ? order.length : order.indexOf(cur);
  setHtml(
    $("progress"),
    order
      .map((sg, i) => {
        const p = stageProgress(a, sg);
        const w = i < curIdx ? 100 : i === curIdx ? (100 * p.done) / p.total : 0;
        return `<div class="seg ${i < curIdx ? "done" : i === curIdx ? "now" : ""}" title="${STAGE_NAME[sg]}"><div class="bar"><i style="width:${w}%"></i></div><div class="name">${STAGE_NAME[sg]}</div></div>`;
      })
      .join(""),
  );

  const cstep = coach?.currentStep;
  if (drill) {
    setHtml($("goal"), `<small>공식 연습</small>${todoText("solve").replace(/^공식 연습: /, "")}`);
    return;
  }
  const stageLine = `${curIdx < order.length ? `${curIdx + 1}/${order.length}단계 · ` : ""}${STAGE_NAME[coach!.currentStage]}`;
  // 공식 이름과 기호는 코치가 말해 준 만큼만 (힌트만 수준: 첫 힌트에 이름, 둘째 힌트 "처음은 …"에 그 수들. 다른 수준은 다)
  const shown = coach!.revealed();
  setHtml(
    $("goal"),
    cstep
      ? `<small>${stageLine}${cstep.caseName && shown.caseName ? `, ${cstep.caseName}` : ""}</small>${lastGoal || holdNote()}${cstep.alg && coach!.level !== "observe" && shown.moves > 0 ? `<code class="alg" title="지금 든 방향 기준, 음성과 같은 기호">${stepSymbols(shown.moves)}</code>` : ""}`
      : coach?.currentStage === "done"
        ? "<small>완성</small>다시 하려면 카드의 버튼을 누르세요."
        : coach?.level === "observe"
          ? `<small>${stageLine}</small>지시 없이, 단계마다 걸린 시간만 재고 있어요.`
          : "<small>목표</small>시작하면 지금 단계와 목표가 여기에 나와요.",
  );
}

function holdNote(): string {
  const g = gripView();
  if (g) return `${iGa(COLOR_KO[g.up])} 위, ${iGa(COLOR_KO[g.front])} 앞에 오게 돌려 잡으세요.`;
  return coach?.currentStep?.view ? "방향 좋아요." : "아무 방향으로 들어도 괜찮아요.";
}

function buildKeys(): void {
  const order = ["i", "k", "j", "f", "h", "g", "d", "e", "s", "l", "w", "o"];
  $("keygrid").innerHTML = order
    .map((k) => {
      const [l, d] = KEYS[k];
      return `<button data-key="${k}" aria-label="${l}${d === 3 ? " 프라임" : ""}, 키 ${k.toUpperCase()}">${l}${d === 3 ? "'" : ""}<kbd>${k.toUpperCase()}</kbd></button>`;
    })
    .join("");
  $("keygrid").querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
    const key = b.dataset.key!;
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      press(key);
    });
    b.addEventListener("pointerup", () => release(key));
    b.addEventListener("pointerleave", () => release(key));
    b.addEventListener("pointercancel", () => release(key));
  });
}

// --- 설정 패널 --------------------------------------------------------------------

const LEVEL_NOTE: Record<string, string> = {
  guided: "한 수씩 불러 주고, 틀리면 다 돌리기 전에 말려 줘요.",
  hint: "혼자 해 보다가 막히면 단계적으로 힌트를 줘요.",
  observe: "지시 없이 돌리는 대로 따라가며 단계별 시간만 재요.",
  drill: "기록상 약한 공식부터, 그 상황을 만들어 알아보기와 실행 시간을 재요.",
};

function levelSummary(): string {
  const m = $<HTMLSelectElement>("method");
  const l = $<HTMLSelectElement>("level");
  const v = $<HTMLSelectElement>("voiceSel");
  const t = (s: HTMLSelectElement) => s.options[s.selectedIndex]?.text ?? "";
  return `${t(m)}, ${t(l)}, ${t(v)} 목소리. 바꾸려면 위의 '설정'.`;
}

function levelNote(): void {
  setText($("levelNote"), LEVEL_NOTE[$<HTMLSelectElement>("level").value] ?? "");
}

function setSetupOpen(on: boolean): void {
  document.body.classList.toggle("setup-open", on);
  const b = $("setupBtn");
  b.classList.toggle("on", on);
  b.setAttribute("aria-expanded", String(on));
}
$("setupBtn").addEventListener("click", () => setSetupOpen(!document.body.classList.contains("setup-open")));
$("known").addEventListener("toggle", () => render());
$("restart").addEventListener("click", () => {
  setSetupOpen(false);
  start();
});

$("scramble").addEventListener("click", () => start());
$("voice").addEventListener("click", () => {
  voice.enabled = !voice.enabled;
  if (!voice.enabled) voice.cancelAll();
  const b = $("voice");
  b.classList.toggle("on", voice.enabled);
  b.setAttribute("aria-pressed", String(voice.enabled));
  b.textContent = voice.enabled ? "음성 켜짐" : "음성 꺼짐";
});
setInterval(() => {
  if (coach) emit(coach.tick(performance.now()));
}, 500);

const voiceSel = $<HTMLSelectElement>("voiceSel");
voiceSel.value = savedVoice() === "M1" ? "M1" : "F2";
voiceSel.addEventListener("change", () => {
  void loadClips(voiceSel.value).then(() => voice.push([{ text: "이 목소리로 안내할게요.", kind: "status", priority: 2 }]));
  render();
});
$("level").addEventListener("change", () => {
  levelBeforeSample = null; // 직접 고른 수준은 그대로 둔다
  levelNote();
  render();
});
$("method").addEventListener("change", render);

setSetupOpen(true);
buildKeys();
levelNote();
render();
void loadClips(voiceSel.value);

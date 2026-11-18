/**
 * 풀이 방법: 지금 어느 단계에 있는지 판별하고, 다음에 할 일(한 덩어리)을 계산한다.
 *
 * 두 가지 방법을 지원한다.
 *  - beginner: 층별 풀이 7단계 (흰 십자 → 1층 꼭짓점 → 2층 → 노란 십자 → 노란 면 → 3층 꼭짓점 → 3층 모서리)
 *  - cfop4   : CFOP 4-look (십자 → F2L → OLL 2단계 → PLL 2단계)
 *  - cfop    : CFOP, PLL 21개를 한 번에 (OLL은 2단계)
 *
 * 흰색으로 시작해 노란색으로 끝낸다. 1·2층을 맞추는 동안에도 노랑을 위로 잡는다
 * (흰 십자가 아래에 오도록) — 처음부터 끝까지 같은 방향이라 덜 헷갈린다.
 *
 * 계획은 항상 "지금 상태"에서 새로 계산한다. 그래서 어떤 상태에서 시작해도, 실수를 해도
 * 다시 이어 갈 수 있다. 한 덩어리를 하는 도중에는 코치가 그 덩어리를 붙들고 있다.
 */

import type { AlgMove, Color, ColorMove, State, View } from "./cube.ts";
import {
  CORNERS,
  EDGES,
  FACELET_POS,
  FACES,
  FORWARD,
  SLOT_NORMAL,
  applyColorMove,
  applyColorMoves,
  centerColor,
  faceColorOfCell,
  findPiece,
  isSolved,
  pieceSolved,
  slotOfColor,
  translate,
  viewColors,
  viewWithRight,
} from "./cube.ts";

export const WHITE = 0;
export const YELLOW = 3;
/** F2L 슬롯 = 이웃한 옆면 두 색 */
export const SLOTS: readonly (readonly [Color, Color])[] = [
  [1, 2],
  [2, 4],
  [4, 5],
  [5, 1],
];
const CROSS: readonly (readonly [Color, Color])[] = [
  [0, 1],
  [0, 2],
  [0, 4],
  [0, 5],
];

export type Method = "beginner" | "cfop4" | "cfop";
export type StageId = "cross" | "corners" | "middle" | "yCross" | "yFace" | "llCorners" | "llEdges" | "f2l" | "oll" | "pll" | "done";

export const STAGES: Record<Method, StageId[]> = {
  beginner: ["cross", "corners", "middle", "yCross", "yFace", "llCorners", "llEdges"],
  cfop4: ["cross", "f2l", "oll", "pll"],
  cfop: ["cross", "f2l", "oll", "pll"],
};

// --- 상태 분석 --------------------------------------------------------------

const solvedPiece = (s: State, cols: readonly Color[]) => pieceSolved(s, findPiece(s, cols));

function inLayer(s: State, cell: number, faceColor: Color): boolean {
  const n = SLOT_NORMAL[slotOfColor(s, faceColor)];
  const p = FACELET_POS[cell];
  return p[0] * n[0] + p[1] * n[1] + p[2] * n[2] === 1;
}

const uTurn = (k: number): ColorMove => ({ color: YELLOW, k: (((k % 4) + 4) % 4) as 1 | 2 | 3 });
const aufState = (s: State, k: number) => (k % 4 ? applyColorMove(s, uTurn(k)) : s);

/** 조각이 놓인 자리가 맞는가 (방향은 무시) */
function pieceInPlace(s: State, piece: readonly number[]): boolean {
  const have = piece.map((i) => s[i]).sort().join();
  const want = piece.map((i) => faceColorOfCell(s, i)).sort().join();
  return have === want;
}

function llPieces(s: State, list: number[][]): number[][] {
  return list.filter((p) => p.some((i) => inLayer(s, i, YELLOW) && faceColorOfCell(s, i) === YELLOW));
}

export interface Analysis {
  cross: number; // 흰 십자 모서리 0~4
  whiteCorners: number; // 1층 꼭짓점 0~4
  middle: number; // 2층 모서리 0~4
  pairs: number; // F2L 슬롯(꼭짓점+모서리) 0~4
  f2l: boolean;
  yEdgesUp: number; // 노란 면을 향한 노란 모서리 스티커 수
  yCornersUp: number;
  oriented: boolean; // 윗면이 모두 노랑
  cornersPermuted: boolean; // 윗면만 돌리면 꼭짓점 자리가 모두 맞는가
  auf: number; // 윗면을 시계 방향으로 몇 번 돌리면 다 맞는가 (-1: 그것만으로는 안 됨)
  solved: boolean;
}

export function analyze(s: State): Analysis {
  const cross = CROSS.filter((e) => solvedPiece(s, e)).length;
  const whiteCorners = SLOTS.filter(([a, b]) => solvedPiece(s, [WHITE, a, b])).length;
  const middle = SLOTS.filter((p) => solvedPiece(s, p)).length;
  const pairs = SLOTS.filter(([a, b]) => solvedPiece(s, [WHITE, a, b]) && solvedPiece(s, [a, b])).length;
  const f2l = cross === 4 && pairs === 4;
  const y = slotOfColor(s, YELLOW) * 9;
  const yEdgesUp = [1, 3, 5, 7].filter((j) => s[y + j] === YELLOW).length;
  const yCornersUp = [0, 2, 6, 8].filter((j) => s[y + j] === YELLOW).length;
  const oriented = yEdgesUp === 4 && yCornersUp === 4;
  let cornersPermuted = false;
  let auf = -1;
  if (f2l) {
    for (let k = 0; k < 4; k++) {
      const t = aufState(s, k);
      if (!cornersPermuted && llPieces(t, CORNERS).every((p) => pieceInPlace(t, p))) cornersPermuted = true;
      if (auf < 0 && isSolved(t)) auf = k;
    }
  }
  return { cross, whiteCorners, middle, pairs, f2l, yEdgesUp, yCornersUp, oriented, cornersPermuted, auf, solved: isSolved(s) };
}

/** 지금 단계: 앞 단계들이 모두 끝난 첫 단계 */
export function stageOf(a: Analysis, method: Method): StageId {
  if (a.solved) return "done";
  if (a.cross < 4) return "cross";
  if (method === "beginner") {
    if (a.whiteCorners < 4) return "corners";
    if (a.middle < 4) return "middle";
    if (a.yEdgesUp < 4) return "yCross";
    if (a.yCornersUp < 4) return "yFace";
    if (!a.cornersPermuted) return "llCorners";
    return "llEdges";
  }
  if (a.pairs < 4) return "f2l";
  if (!a.oriented) return "oll";
  return "pll";
}

/** 단계 안에서의 진행도 0~1 (화면 표시·음성 안내용) */
export function stageProgress(a: Analysis, stage: StageId): { done: number; total: number } {
  switch (stage) {
    case "cross":
      return { done: a.cross, total: 4 };
    case "corners":
      return { done: a.whiteCorners, total: 4 };
    case "middle":
      return { done: a.middle, total: 4 };
    case "f2l":
      return { done: a.pairs, total: 4 };
    case "yCross":
      return { done: a.yEdgesUp, total: 4 };
    case "yFace":
      return { done: a.yCornersUp, total: 4 };
    case "oll":
      return { done: a.yEdgesUp + a.yCornersUp, total: 8 };
    default:
      return { done: a.solved ? 1 : 0, total: 1 };
  }
}

// --- 조각 단위 탐색 ---------------------------------------------------------
// 몇 개 조각의 위치만 따라가며 탐색한다. 조각 하나의 상태(자리+방향)는 그 조각의 기준
// 스티커 하나가 어느 칸에 있는지로 정해진다 (모서리 24가지, 꼭짓점 24가지).

const EDGE_CELLS = EDGES.flat().sort((a, b) => a - b);
const CORNER_CELLS = CORNERS.flat().sort((a, b) => a - b);
const IS_EDGE_CELL = new Set(EDGE_CELLS);
const LOCAL = new Int16Array(54).fill(-1);
EDGE_CELLS.forEach((c, i) => (LOCAL[c] = i));
CORNER_CELLS.forEach((c, i) => (LOCAL[c] = i));

function encode(cells: readonly number[]): number {
  let idx = 0;
  for (let i = cells.length - 1; i >= 0; i--) idx = idx * 24 + LOCAL[cells[i]];
  return idx;
}

/** 기준 스티커가 지금 있는 칸 */
function currentCell(s: State, colors: readonly Color[], ref: Color): number {
  return findPiece(s, colors).find((i) => s[i] === ref)!;
}

/** 기준 스티커가 맞춰졌을 때 있어야 할 칸 */
function homeCell(s: State, colors: readonly Color[], ref: Color): number {
  const want = [...colors].sort().join();
  for (const p of colors.length === 2 ? EDGES : CORNERS) {
    if (p.map((i) => faceColorOfCell(s, i)).sort().join() === want) return p.find((i) => faceColorOfCell(s, i) === ref)!;
  }
  throw new Error("자리를 찾을 수 없음");
}

/** 탐색에 쓰는 회전: 슬롯 기호. exclude에 든 가운데 색의 면은 돌리지 않는다. */
function slotMoves(s: State, exclude: Color[] = [], prefer: Color[] = []): string[] {
  const slots = [0, 1, 2, 3, 4, 5].filter((sl) => !exclude.includes(centerColor(s, sl)));
  const rank = (sl: number) => {
    const i = prefer.indexOf(centerColor(s, sl));
    return i < 0 ? prefer.length : i;
  };
  slots.sort((a, b) => rank(a) - rank(b));
  return slots.flatMap((sl) => ["", "'", "2"].map((x) => FACES[sl] + x));
}

const toColorMove = (s: State, m: string): ColorMove => ({
  color: centerColor(s, FACES.indexOf(m[0])),
  k: m.length === 1 ? 1 : m[1] === "2" ? 2 : 3,
});

const tableCache = new Map<string, Uint8Array>();

/** 목표 칸들에서 시작하는 너비 우선 탐색 거리표 (조각 4개까지) */
function distanceTable(goal: readonly number[], moves: readonly string[]): Uint8Array {
  const key = goal.join(",") + "|" + moves.join(" ");
  const hit = tableCache.get(key);
  if (hit) return hit;
  const k = goal.length;
  const size = 24 ** k;
  const types = goal.map((c) => (IS_EDGE_CELL.has(c) ? EDGE_CELLS : CORNER_CELLS));
  const dist = new Uint8Array(size).fill(255);
  const queue = new Int32Array(size);
  const start = encode(goal);
  dist[start] = 0;
  queue[0] = start;
  let head = 0;
  let tail = 1;
  const cells = new Array<number>(k);
  const fwd = moves.map((m) => FORWARD[m]);
  while (head < tail) {
    let idx = queue[head++];
    const d = dist[idx];
    for (let i = 0; i < k; i++) {
      cells[i] = types[i][idx % 24];
      idx = Math.floor(idx / 24);
    }
    for (const f of fwd) {
      let n = 0;
      for (let i = k - 1; i >= 0; i--) n = n * 24 + LOCAL[f[cells[i]]];
      if (dist[n] === 255) {
        dist[n] = d + 1;
        queue[tail++] = n;
      }
    }
  }
  tableCache.set(key, dist);
  return dist;
}

/** 거리표를 따라 내려가며 해법을 만든다 */
function descend(start: number[], goal: number[], moves: string[]): string[] {
  const table = distanceTable(goal, moves);
  const out: string[] = [];
  let cells = start;
  let d = table[encode(cells)];
  while (d > 0) {
    for (const m of moves) {
      const next = cells.map((c) => FORWARD[m][c]);
      if (table[encode(next)] === d - 1) {
        out.push(m);
        cells = next;
        d--;
        break;
      }
    }
  }
  return out;
}

const SAME_AXIS: Record<string, string> = { U: "D", D: "U", R: "L", L: "R", F: "B", B: "F" };

/** 반복 깊이 A* (F2L 한 쌍 넣기처럼 조각이 많아 거리표를 통째로 못 만들 때) */
function idaStar(start: number[], heuristic: (cells: number[]) => number, moves: string[], maxDepth: number, nodeLimit: number): string[] | null {
  const path: string[] = [];
  let nodes = 0;
  const FOUND = -1;
  const dfs = (cells: number[], g: number, bound: number, last: string): number => {
    const h = heuristic(cells);
    if (g + h > bound) return g + h;
    if (h === 0) return FOUND;
    let min = Infinity;
    for (const m of moves) {
      const face = m[0];
      if (face === last) continue;
      if (SAME_AXIS[face] === last && FACES.indexOf(face) < FACES.indexOf(last)) continue;
      if (++nodes > nodeLimit) throw new Error("budget");
      path.push(m);
      const t = dfs(
        cells.map((c) => FORWARD[m][c]),
        g + 1,
        bound,
        face,
      );
      if (t === FOUND) return FOUND;
      path.pop();
      if (t < min) min = t;
    }
    return min;
  };
  try {
    let bound = heuristic(start);
    while (bound <= maxDepth) {
      const t = dfs(start, 0, bound, "");
      if (t === FOUND) return [...path];
      bound = t;
    }
  } catch {
    return null;
  }
  return null;
}

// --- 계획 ------------------------------------------------------------------

/** 다음에 할 한 덩어리 */
export interface Step {
  stage: StageId;
  kind: "search" | "procedure" | "alg" | "auf";
  /** 무엇을 하려는지: 말로 안내할 때 쓰는 구조화된 정보 */
  intent: Intent;
  /** 이 방향으로 잡고 하기를 권한다 (없으면 아무 방향이나 괜찮다) */
  view?: View;
  alg?: string; // 공식 기호 (view 기준)
  caseName?: string;
  repeat?: number;
  moves: AlgMove[];
}

export type Intent =
  | { type: "crossEdge"; colors: [Color, Color] }
  | { type: "crossAll" }
  | { type: "cornerOut"; colors: [Color, Color, Color] }
  | { type: "cornerAbove"; colors: [Color, Color, Color] }
  | { type: "cornerInsert"; colors: [Color, Color, Color] }
  | { type: "edgeAlign"; colors: [Color, Color] }
  | { type: "edgeInsert"; colors: [Color, Color]; side: "right" | "left" }
  | { type: "edgeOut"; colors: [Color, Color] }
  | { type: "pair"; colors: [Color, Color] }
  | { type: "llAlg"; shape: string }
  | { type: "auf"; final: boolean };

const DEFAULT_VIEW = (front: Color): View => ({ up: YELLOW, front });

/** 색 회전열을 보는 방향 v의 기호가 붙은 AlgMove로 */
function withView(s: State, ms: readonly ColorMove[], v: View): AlgMove[] {
  const vc = viewColors(s, v);
  const letter = (c: Color) => (Object.keys(vc) as (keyof typeof vc)[]).find((l) => vc[l] === c)!;
  return ms.map((m) => ({ ...m, token: letter(m.color) + ["", "", "2", "'"][m.k], viewBefore: v }));
}

/** 슬롯(두 옆색)이 앞-오른쪽에 오도록, 노랑을 위로 잡는 방향 */
function slotView(s: State, a: Color, b: Color): View {
  const v1 = viewWithRight(s, YELLOW, a);
  return v1.front === b ? v1 : viewWithRight(s, YELLOW, b);
}

// 1단계: 흰 십자 --------------------------------------------------------------

function planCross(s: State, allAtOnce: boolean): Step {
  const moves = slotMoves(s, [], [YELLOW, 1, 2, 4, 5, WHITE]);
  const solved = CROSS.filter((e) => solvedPiece(s, e));
  if (allAtOnce) {
    const start = CROSS.map((e) => currentCell(s, e, WHITE));
    const goal = CROSS.map((e) => homeCell(s, e, WHITE));
    const sol = descend(start, goal, moves).map((m) => toColorMove(s, m));
    return { stage: "cross", kind: "search", intent: { type: "crossAll" }, moves: withView(s, sol, DEFAULT_VIEW(2)) };
  }
  let best: { e: readonly [Color, Color]; sol: string[] } | null = null;
  for (const e of CROSS) {
    if (solved.includes(e)) continue;
    const set = [...solved, e];
    const sol = descend(
      set.map((x) => currentCell(s, x, WHITE)),
      set.map((x) => homeCell(s, x, WHITE)),
      moves,
    );
    if (!best || sol.length < best.sol.length) best = { e, sol };
  }
  const { e, sol } = best!;
  return {
    stage: "cross",
    kind: "search",
    intent: { type: "crossEdge", colors: [e[0], e[1]] },
    moves: withView(
      s,
      sol.map((m) => toColorMove(s, m)),
      DEFAULT_VIEW(e[1]),
    ),
  };
}

// 2단계: 1층 꼭짓점 (섹시무브 반복) ---------------------------------------------

const SEXY = "R U R' U'";

function planWhiteCorner(s: State): Step {
  type Option = { cost: number; step: Step };
  const options: Option[] = [];
  for (const [a, b] of SLOTS) {
    const colors: [Color, Color, Color] = [WHITE, a, b];
    if (solvedPiece(s, colors)) continue;
    const piece = findPiece(s, colors);
    const up = piece.some((i) => inLayer(s, i, YELLOW));
    if (!up) {
      // 아래층 엉뚱한 자리(또는 제자리에서 돌아간 채)에 있다: 그 자리를 앞-오른쪽에 두고 한 번 꺼낸다
      const [c1, c2] = piece.map((i) => faceColorOfCell(s, i)).filter((c) => c !== WHITE);
      const view = slotView(s, c1, c2);
      options.push({
        cost: 20,
        step: { stage: "corners", kind: "procedure", intent: { type: "cornerOut", colors }, view, alg: SEXY, moves: translate(s, SEXY, view) },
      });
      continue;
    }
    for (let k = 0; k < 4; k++) {
      const t = aufState(s, k);
      const p = findPiece(t, colors);
      const sides = p.map((i) => faceColorOfCell(t, i)).filter((c) => c !== YELLOW);
      if (!(sides.includes(a) && sides.includes(b))) continue;
      // 자리 위에 오면 섹시무브를 몇 번 해야 들어가는지 (1, 3, 5번)
      const view = slotView(t, a, b);
      let u = t;
      let reps = 0;
      while (!solvedPiece(u, colors) && reps < 6) {
        u = applyColorMoves(u, translate(u, SEXY, view));
        reps++;
      }
      // 윗면 돌리기의 비용에도 그 뒤 섹시무브 횟수를 넣어야, 돌린 다음에 다른 꼭짓점으로 갈아타지 않는다
      if (k) {
        options.push({
          cost: 1 + reps * 4,
          step: { stage: "corners", kind: "auf", intent: { type: "cornerAbove", colors }, moves: withView(s, [uTurn(k)], view) },
        });
      } else {
        const moves = Array.from({ length: reps }, () => translate(s, SEXY, view)).flat();
        options.push({
          cost: reps * 4,
          step: { stage: "corners", kind: "procedure", intent: { type: "cornerInsert", colors }, view, alg: SEXY, repeat: reps, moves },
        });
      }
      break;
    }
  }
  options.sort((x, y) => x.cost - y.cost);
  return options[0].step;
}

// 3단계: 2층 모서리 ----------------------------------------------------------

export const RIGHT_INSERT = "U R U' R' U' F' U F";
export const LEFT_INSERT = "U' L' U L U F U' F'";

function planMiddleEdge(s: State): Step {
  let best: { k: number; step: Step } | null = null;
  for (const p of EDGES) {
    const cols = p.map((i) => s[i]);
    if (cols.includes(YELLOW) || !p.some((i) => inLayer(s, i, YELLOW))) continue;
    const colors: [Color, Color] = [cols[0], cols[1]];
    for (let k = 0; k < 4; k++) {
      const t = aufState(s, k);
      const q = findPiece(t, colors);
      const side = q.find((i) => faceColorOfCell(t, i) !== YELLOW)!;
      const top = q.find((i) => faceColorOfCell(t, i) === YELLOW)!;
      if (t[side] !== faceColorOfCell(t, side)) continue;
      if (best && best.k <= (k ? 1 : 0)) break;
      if (k) {
        best = { k: 1, step: { stage: "middle", kind: "auf", intent: { type: "edgeAlign", colors }, moves: withView(s, [uTurn(k)], DEFAULT_VIEW(t[side])) } };
      } else {
        const view = DEFAULT_VIEW(t[side]);
        const vc = viewColors(s, view);
        const right = t[top] === vc.R;
        const alg = right ? RIGHT_INSERT : LEFT_INSERT;
        best = {
          k: 0,
          step: { stage: "middle", kind: "alg", intent: { type: "edgeInsert", colors, side: right ? "right" : "left" }, view, alg, moves: translate(s, alg, view) },
        };
      }
      break;
    }
  }
  if (best) return best.step;
  // 위층에 넣을 모서리가 없다: 잘못 들어간 2층 모서리를 꺼낸다
  const slot = SLOTS.find((pr) => !solvedPiece(s, pr))!;
  const view = slotView(s, slot[0], slot[1]);
  const occupant = EDGES.find((p) => p.map((i) => faceColorOfCell(s, i)).sort().join() === [...slot].sort().join())!;
  const colors = occupant.map((i) => s[i]) as [Color, Color];
  return { stage: "middle", kind: "alg", intent: { type: "edgeOut", colors }, view, alg: RIGHT_INSERT, moves: translate(s, RIGHT_INSERT, view) };
}

// 노란 면 쪽: 공식 고르기 -----------------------------------------------------

export interface Alg {
  name: string;
  alg: string;
}

/**
 * 보는 방향(노랑 위, 앞 색 4가지)과 윗면 미리 돌리기(0~3)와 공식의 조합을 최대 maxApps번까지
 * 이어 붙여 goal에 닿는 가장 짧은 길을 찾고, 그 첫 수를 돌려준다.
 * 공식을 상황별로 외워 둔 규칙("ㄱ자는 왼쪽 뒤에")을 코드로 옮겨 적는 대신 직접 시험해 보므로,
 * 규칙을 잘못 옮겨 적을 일이 없다.
 */
function searchAlgs(
  s: State,
  algs: readonly Alg[],
  goal: (t: State) => boolean,
  maxApps: number,
  allowAUF: boolean,
): { view: View; auf: number; alg: Alg } | null {
  type Act = { view: View; auf: number; alg: Alg };
  const views = [2, 1, 5, 4].map(DEFAULT_VIEW);
  let frontier: { s: State; first: Act | null }[] = [{ s, first: null }];
  for (let depth = 1; depth <= maxApps; depth++) {
    let best: { act: Act; cost: number } | null = null;
    const next: typeof frontier = [];
    const seen = new Set<string>();
    for (const node of frontier) {
      for (const auf of allowAUF ? [0, 1, 3, 2] : [0]) {
        const s1 = aufState(node.s, auf);
        for (const view of views) {
          for (const alg of algs) {
            const s2 = applyColorMoves(s1, translate(s1, alg.alg, view));
            const act = node.first ?? { view, auf, alg };
            if (goal(s2)) {
              const cost = (act.auf ? 100 : 0) + act.alg.alg.split(" ").length;
              if (!best || cost < best.cost) best = { act, cost };
            } else if (depth < maxApps) {
              const key = s2.join("");
              if (!seen.has(key)) {
                seen.add(key);
                next.push({ s: s2, first: act });
              }
            }
          }
        }
      }
    }
    if (best) return best.act;
    frontier = next;
  }
  return null;
}

function algStep(s: State, stage: StageId, found: { view: View; auf: number; alg: Alg }, shape: string): Step {
  if (found.auf) {
    return { stage, kind: "auf", intent: { type: "auf", final: false }, moves: withView(s, [uTurn(found.auf)], found.view) };
  }
  return {
    stage,
    kind: "alg",
    intent: { type: "llAlg", shape },
    view: found.view,
    alg: found.alg.alg,
    caseName: found.alg.name,
    moves: translate(s, found.alg.alg, found.view),
  };
}

/** 윗면 노란 모서리 모양: 점 / ㄱ자 / 일자 / 십자 */
export function edgeShape(s: State): string {
  const y = slotOfColor(s, YELLOW) * 9;
  const up = [1, 3, 5, 7].filter((j) => s[y + j] === YELLOW);
  if (up.length === 4) return "십자";
  if (up.length === 0) return "점";
  if (up.length === 2) return (up[0] === 1 && up[1] === 7) || (up[0] === 3 && up[1] === 5) ? "일자" : "ㄱ자";
  return `${up.length}개`;
}

export function cornerShape(s: State): string {
  const y = slotOfColor(s, YELLOW) * 9;
  const n = [0, 2, 6, 8].filter((j) => s[y + j] === YELLOW).length;
  // 숫자는 읽는 그대로 적는다 (음성 합성기가 "2개"를 "이 개"로 읽지 않게)
  return n === 1 ? "물고기" : n === 0 ? "노란 꼭짓점이 없는" : `노란 꼭짓점 ${["", "한", "두", "세", "네"][n]} 개`;
}

export const BEGINNER_ALGS = {
  yCross: { name: "노란 십자 공식", alg: "F R U R' U' F'" },
  sune: { name: "수네", alg: "R U R' U R U2 R'" },
  corners: { name: "꼭짓점 자리 바꾸기", alg: "R' F R' B2 R F' R' B2 R2" },
  edges: { name: "모서리 돌리기", alg: "R U' R U R U R U' R' U' R2" },
} as const;

export const OLL_EDGES: Alg[] = [
  { name: "에프 섹시", alg: "F R U R' U' F'" },
  { name: "에프 리버스 섹시", alg: "F U R U' R' F'" },
];

/** 꼭짓점 방향 맞추기 (OCLL) 7가지 */
export const OCLL: Alg[] = [
  { name: "수네", alg: "R U R' U R U2 R'" },
  { name: "안티수네", alg: "R U2 R' U' R U' R'" },
  { name: "H", alg: "R U R' U R U' R' U R U2 R'" },
  { name: "파이", alg: "R U2 R2 U' R2 U' R2 U2 R" },
  { name: "헤드라이트", alg: "R2 D R' U2 R D' R' U2 R'" },
  { name: "T", alg: "r U R' U' r' F R F'" },
  { name: "나비", alg: "F R' F' r U R U' r'" },
];

/** PLL 21가지. 가운데 줄(M) 돌리기가 없는 공식만 쓴다 (추적기가 한 층 회전만 알아본다). */
export const PLL: Alg[] = [
  { name: "Aa", alg: "R' F R' B2 R F' R' B2 R2" },
  { name: "Ab", alg: "R B' R F2 R' B R F2 R2" },
  { name: "E", alg: "x' R U' R' D R U R' D' R U R' D R U' R' D' x" },
  { name: "F", alg: "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R" },
  { name: "Ga", alg: "R2 U R' U R' U' R U' R2 U' D R' U R D'" },
  { name: "Gb", alg: "R' U' R U D' R2 U R' U R U' R U' R2 D" },
  { name: "Gc", alg: "R2 U' R U' R U R' U R2 U D' R U' R' D" },
  { name: "Gd", alg: "R U R' U' D R2 U' R U' R' U R' U R2 D'" },
  { name: "H", alg: "R2 U2 R U2 R2 U2 R2 U2 R U2 R2" },
  { name: "Ja", alg: "R' U L' U2 R U' R' U2 R L" },
  { name: "Jb", alg: "R U R' F' R U R' U' R' F R2 U' R'" },
  { name: "Na", alg: "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'" },
  { name: "Nb", alg: "R' U R U' R' F' U' F R U R' F R' F' R U' R" },
  { name: "Ra", alg: "R U' R' U' R U R D R' U' R D' R' U2 R'" },
  { name: "Rb", alg: "R2 F R U R U' R' F' R U2 R' U2 R" },
  { name: "T", alg: "R U R' U' R' F R2 U' R' U' R U R' F'" },
  { name: "Ua", alg: "R U' R U R U R U' R' U' R2" },
  { name: "Ub", alg: "R2 U R U R' U' R' U' R' U R'" },
  { name: "V", alg: "R' U R' d' R' F' R2 U' R' U R' F R F" },
  { name: "Y", alg: "F R U' R' U' R U R' F' R U R' U' R' F R F'" },
  { name: "Z", alg: "R' U' R U' R U R U' R' U R U R2 U' R'" },
];

const pll = (name: string) => PLL.find((p) => p.name === name)!;

function planLastLayer(s: State, a: Analysis, method: Method): Step {
  const stage = stageOf(a, method);
  const solvedUpToAuf = (t: State) => analyze(t).auf >= 0;
  if (method === "beginner") {
    if (stage === "yCross") {
      const f = searchAlgs(s, [BEGINNER_ALGS.yCross], (t) => analyze(t).yEdgesUp === 4, 3, false)!;
      return algStep(s, stage, f, edgeShape(s));
    }
    if (stage === "yFace") {
      const f = searchAlgs(s, [BEGINNER_ALGS.sune], (t) => analyze(t).oriented, 3, false)!;
      return algStep(s, stage, f, cornerShape(s));
    }
    if (stage === "llCorners") {
      const f = searchAlgs(s, [BEGINNER_ALGS.corners], (t) => analyze(t).cornersPermuted, 2, true)!;
      return algStep(s, stage, f, "꼭짓점");
    }
    if (a.auf > 0) return finalAuf(s, a, "llEdges");
    const f = searchAlgs(s, [BEGINNER_ALGS.edges], solvedUpToAuf, 3, true)!;
    return algStep(s, "llEdges", f, "모서리");
  }
  if (stage === "oll") {
    if (a.yEdgesUp < 4) {
      const f = searchAlgs(s, OLL_EDGES, (t) => analyze(t).yEdgesUp === 4, 2, false)!;
      return algStep(s, stage, f, edgeShape(s));
    }
    const f = searchAlgs(s, OCLL, (t) => analyze(t).oriented, 1, false)!;
    return algStep(s, stage, f, cornerShape(s));
  }
  if (a.auf > 0) return finalAuf(s, a, "pll");
  if (method === "cfop4" && !a.cornersPermuted) {
    const f = searchAlgs(s, [pll("T"), pll("Y")], (t) => analyze(t).cornersPermuted, 1, true)!;
    return algStep(s, stage, f, "꼭짓점");
  }
  const set = method === "cfop4" ? ["Ua", "Ub", "H", "Z"].map(pll) : PLL;
  const f = searchAlgs(s, set, solvedUpToAuf, 1, true)!;
  return algStep(s, stage, f, "모서리");
}

function finalAuf(s: State, a: Analysis, stage: StageId): Step {
  return { stage, kind: "auf", intent: { type: "auf", final: true }, moves: withView(s, [uTurn(a.auf)], DEFAULT_VIEW(2)) };
}

// F2L 한 쌍 ---------------------------------------------------------------

function planPair(s: State): Step | null {
  const moves = slotMoves(s, [WHITE], [YELLOW, 1, 2, 4, 5]);
  const crossGoal = CROSS.map((e) => homeCell(s, e, WHITE));
  const crossTable = distanceTable(crossGoal, moves);
  const solvedSlots = SLOTS.filter(([a, b]) => solvedPiece(s, [WHITE, a, b]) && solvedPiece(s, [a, b]));
  let best: { slot: readonly [Color, Color]; sol: string[] } | null = null;
  for (const slot of SLOTS) {
    if (solvedSlots.includes(slot)) continue;
    const groups = [...solvedSlots, slot];
    const start = [
      ...CROSS.map((e) => currentCell(s, e, WHITE)),
      ...groups.flatMap(([a, b]) => [currentCell(s, [WHITE, a, b], WHITE), currentCell(s, [a, b], a)]),
    ];
    const goal = [...crossGoal, ...groups.flatMap(([a, b]) => [homeCell(s, [WHITE, a, b], WHITE), homeCell(s, [a, b], a)])];
    const pairTables = groups.map((_, g) => distanceTable(goal.slice(4 + 2 * g, 6 + 2 * g), moves));
    const h = (cells: number[]) => {
      let m = crossTable[encode(cells.slice(0, 4))];
      for (let g = 0; g < pairTables.length; g++) m = Math.max(m, pairTables[g][encode(cells.slice(4 + 2 * g, 6 + 2 * g))]);
      return m;
    };
    const limit = best ? best.sol.length : 12;
    const sol = idaStar(start, h, moves, limit, 400_000);
    if (sol && (!best || sol.length < best.sol.length)) best = { slot, sol };
  }
  if (!best) return null;
  const [a, b] = best.slot;
  const view = slotView(s, a, b);
  return {
    stage: "f2l",
    kind: "search",
    intent: { type: "pair", colors: [a, b] },
    view,
    moves: withView(
      s,
      best.sol.map((m) => toColorMove(s, m)),
      view,
    ),
  };
}


/** 연습 모드·테스트용: 공식 하나로 goal에 닿는 보는 방향과 윗면 돌리기 */
export function searchAlgsPublic(s: State, algs: readonly Alg[], goal: (t: State) => boolean) {
  return searchAlgs(s, algs, goal, 1, true);
}

// --- 진입점 -----------------------------------------------------------------

export function planNext(s: State, method: Method): Step | null {
  const a = analyze(s);
  const stage = stageOf(a, method);
  switch (stage) {
    case "done":
      return null;
    case "cross":
      return planCross(s, method !== "beginner");
    case "corners":
      return planWhiteCorner(s);
    case "middle":
      return planMiddleEdge(s);
    case "f2l": {
      const step = planPair(s);
      if (step) return step;
      // 탐색이 예산 안에 못 끝나면 초급 방식(꼭짓점 → 모서리)으로 넣는다
      return a.whiteCorners < 4 ? { ...planWhiteCorner(s), stage: "f2l" } : { ...planMiddleEdge(s), stage: "f2l" };
    }
    default:
      return planLastLayer(s, a, method);
  }
}

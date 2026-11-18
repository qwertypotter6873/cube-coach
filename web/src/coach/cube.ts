/**
 * 코치 엔진의 큐브 모델.
 *
 * 상태는 54칸의 색 번호 배열이다. 칸 순서는 파이썬 엔진(cube.py)과 같다:
 * 면 슬롯 U R F D L B 순서로 9칸씩, 각 면을 정면에서 볼 때 왼쪽 위부터 행 우선.
 * 색 번호: 0 흰 1 빨 2 초 3 노 4 주 5 파.
 *
 * 중요한 점: 추적기는 큐브를 통째로 돌린 것까지 상태에 반영하므로, 가운데 칸 색이
 * 어느 슬롯에 있는지는 고정되어 있지 않다. 그래서 코치는 면을 슬롯 이름이 아니라
 * "가운데 색"으로 부른다. 공식(R U R' U' …)은 "보는 방향"(위 색, 앞 색)을 정하면
 * 가운데 색 회전으로 번역된다.
 */

import tables from "./tables.json" with { type: "json" };

export type State = Uint8Array;
export type Color = number;
export type Vec3 = readonly [number, number, number];

export const FACES = tables.faces; // "URFDLB"
export const COLOR_KO = ["흰색", "빨강", "초록", "노랑", "주황", "파랑"] as const;
export const COLOR_SHORT = ["흰", "빨", "초", "노", "주", "파"] as const;
export const SOLVED: State = Uint8Array.from(tables.solved);

const MOVE_PERMS: Record<string, Int16Array> = Object.fromEntries(
  Object.entries(tables.moves).map(([k, v]) => [k, Int16Array.from(v as number[])]),
);
export const FACELET_POS: Vec3[] = tables.facelets.map((f) => f.pos as unknown as Vec3);
export const FACELET_NORMAL: Vec3[] = tables.facelets.map((f) => f.normal as unknown as Vec3);
/** 슬롯(0~5, URFDLB)의 바깥 방향 */
export const SLOT_NORMAL: Vec3[] = [0, 1, 2, 3, 4, 5].map((s) => FACELET_NORMAL[s * 9 + 4]);

// --- 기본 연산 ---------------------------------------------------------------

/** 슬롯 기호 회전 하나(예: "R", "U2", "x'")를 적용한 새 상태 */
export function applySlotMove(s: State, move: string): State {
  const p = MOVE_PERMS[move];
  if (!p) throw new Error(`알 수 없는 회전: ${move}`);
  const out = new Uint8Array(54);
  for (let i = 0; i < 54; i++) out[i] = s[p[i]];
  return out;
}

export function applySlotAlg(s: State, alg: string): State {
  for (const m of alg.split(/\s+/).filter(Boolean)) s = applySlotMove(s, m);
  return s;
}

/** 칸 i에 있던 스티커가 회전 뒤 가는 자리 (forward[move][i]) */
export const FORWARD: Record<string, Int16Array> = Object.fromEntries(
  Object.entries(MOVE_PERMS).map(([k, p]) => {
    const f = new Int16Array(54);
    for (let i = 0; i < 54; i++) f[p[i]] = i;
    return [k, f];
  }),
);

export const centerColor = (s: State, slot: number): Color => s[slot * 9 + 4];

export function slotOfColor(s: State, c: Color): number {
  for (let slot = 0; slot < 6; slot++) if (s[slot * 9 + 4] === c) return slot;
  throw new Error(`가운데 색 ${c}를 찾을 수 없음`);
}

export const opposite = (c: Color): Color => [3, 4, 5, 0, 1, 2][c];

// --- 색 기준 회전 -------------------------------------------------------------

/** 가운데 색이 c인 층을 시계 방향으로 k번(90도 단위) 돌린다. 추적기가 보고하는 단위와 같다. */
export interface ColorMove {
  color: Color;
  k: 1 | 2 | 3;
}

const SUFFIX = { 1: "", 2: "2", 3: "'" } as const;

export function applyColorMove(s: State, m: ColorMove): State {
  return applySlotMove(s, FACES[slotOfColor(s, m.color)] + SUFFIX[m.k]);
}

export function applyColorMoves(s: State, ms: readonly ColorMove[]): State {
  for (const m of ms) s = applyColorMove(s, m);
  return s;
}

export const inverse = (m: ColorMove): ColorMove => ({ color: m.color, k: (4 - m.k) as 1 | 2 | 3 });

// --- 보는 방향(view) ----------------------------------------------------------

const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const eq = (a: Vec3, b: Vec3) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
const neg = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];

/** 사람이 큐브를 보는 방향: 위를 향한 가운데 색과 앞(자기 쪽)을 향한 가운데 색 */
export interface View {
  up: Color;
  front: Color;
}

export type Letter = "U" | "R" | "F" | "D" | "L" | "B";

/** 보는 방향에서 각 면 기호(U R F D L B)가 어느 가운데 색인지 */
export function viewColors(s: State, v: View): Record<Letter, Color> {
  const nu = SLOT_NORMAL[slotOfColor(s, v.up)];
  const nf = SLOT_NORMAL[slotOfColor(s, v.front)];
  const nr = cross(nu, nf); // 오른쪽 = 위 × 앞
  const colorAt = (n: Vec3): Color => centerColor(s, SLOT_NORMAL.findIndex((x) => eq(x, n)));
  return { U: v.up, F: v.front, R: colorAt(nr), D: colorAt(neg(nu)), B: colorAt(neg(nf)), L: colorAt(neg(nr)) };
}

/** 그 보는 방향에서 색이 c인 면의 기호 */
export function letterOf(s: State, v: View, c: Color): Letter {
  const vc = viewColors(s, v);
  return (Object.keys(vc) as Letter[]).find((l) => vc[l] === c)!;
}

export function isValidView(v: View): boolean {
  return v.up !== v.front && opposite(v.up) !== v.front;
}

export const ALL_VIEWS: View[] = [0, 1, 2, 3, 4, 5].flatMap((up) =>
  [0, 1, 2, 3, 4, 5].filter((front) => front !== up && front !== opposite(up)).map((front) => ({ up, front })),
);

/**
 * 보는 방향이 from 에서 to 로 바뀐 것과 똑같이 큐브를 돌렸을 때, 방향 v 로 보던 사람은 어떤 방향이 되는가.
 * (공식의 두 줄 돌리기가 큐브를 통째로 돌릴 때 쓴다: from→to 는 공식 기준, v 는 실제로 들고 있는 방향)
 */
export function rebaseView(s: State, from: View, to: View, v: View): View {
  if (from.up === to.up && from.front === to.front) return v;
  const a = viewColors(s, from);
  const b = viewColors(s, to);
  const map = (c: Color): Color => b[(Object.keys(a) as Letter[]).find((l) => a[l] === c)!];
  return { up: map(v.up), front: map(v.front) };
}

/** 위 색과 오른쪽 색으로 보는 방향 정하기 (예: 흰 아래, 슬롯이 앞-오른쪽) */
export function viewWithRight(s: State, up: Color, right: Color): View {
  for (const v of ALL_VIEWS) if (v.up === up && viewColors(s, v).R === right) return v;
  throw new Error("불가능한 방향");
}

// --- 공식 표기 ---------------------------------------------------------------

/**
 * 공식의 한 수. token은 공식에 적힌 그대로의 기호(R, U', r, …)이고,
 * color/k는 실제로 돌아가는 층(가운데 색 기준)과 방향이다.
 * 두 줄 돌리기 r = L x 이므로, r의 color는 그 방향에서 왼쪽 면의 색이다
 * (가운데 색 기준으로는 왼쪽 층이 돈 것과 같고, 큐브 전체가 함께 돈다).
 */
export interface AlgMove extends ColorMove {
  token: string;
  viewBefore: View; // 이 수를 둘 때 사람이 보는 방향 (통째 회전·두 줄 돌리기 뒤에는 바뀐다)
}

const K_OF: Record<string, 1 | 2 | 3> = { "": 1, "2": 2, "'": 3, "2'": 2 };
const OPP_LETTER: Record<Letter, Letter> = { U: "D", D: "U", R: "L", L: "R", F: "B", B: "F" };
/** 두 줄 돌리기 = 반대 면 + 통째 회전.  Rw = L x, Lw = R x', Uw = D y, Dw = U y', Fw = B z, Bw = F z' */
const WIDE_ROT: Record<Letter, [string, 1 | -1]> = { R: ["x", 1], L: ["x", -1], U: ["y", 1], D: ["y", -1], F: ["z", 1], B: ["z", -1] };

/** 큐브를 통째로 돌린 뒤의 보는 방향. x: 앞이 위로, y: 오른쪽이 앞으로, z: 왼쪽이 위로 */
export function rotateView(s: State, v: View, axis: string, k: number): View {
  for (let i = 0; i < ((k % 4) + 4) % 4; i++) {
    const vc = viewColors(s, v);
    v = axis === "x" ? { up: vc.F, front: vc.D } : axis === "y" ? { up: vc.U, front: vc.R } : { up: vc.L, front: vc.F };
  }
  return v;
}

function parseToken(tok: string): { base: string; k: 1 | 2 | 3 } {
  const m = /^([URFDLBurfdlbxyzMES])(w?)(2'|2|')?$/.exec(tok);
  if (!m) throw new Error(`공식에 쓸 수 없는 기호: ${tok}`);
  const base = m[2] ? m[1].toLowerCase() : m[1];
  return { base, k: K_OF[m[3] ?? ""] };
}

/** 공식 문자열을 보는 방향 v에서 시작했을 때의 색 회전열로 번역한다. 통째 회전(x y z)은 보는 방향만 바꾼다. */
export function translate(s: State, alg: string, v: View): AlgMove[] {
  const out: AlgMove[] = [];
  let view = v;
  for (const tok of alg.split(/\s+/).filter(Boolean)) {
    const { base, k } = parseToken(tok);
    const vc = viewColors(s, view);
    if ("URFDLB".includes(base)) {
      out.push({ token: tok, color: vc[base as Letter], k, viewBefore: view });
    } else if ("urfdlb".includes(base)) {
      const face = base.toUpperCase() as Letter;
      out.push({ token: tok, color: vc[OPP_LETTER[face]], k, viewBefore: view });
      const [axis, sign] = WIDE_ROT[face];
      view = rotateView(s, view, axis, sign * k);
    } else if ("xyz".includes(base)) {
      view = rotateView(s, view, base, k);
    } else {
      throw new Error(`가운데 줄 돌리기(${tok})는 아직 지원하지 않음`);
    }
  }
  return out;
}

export function algToString(ms: readonly AlgMove[]): string {
  return ms.map((m) => m.token).join(" ");
}

export function invertAlg(alg: string): string {
  return alg
    .split(/\s+/)
    .filter(Boolean)
    .reverse()
    .map((t) => (t.endsWith("2'") ? t.slice(0, -1) : t.endsWith("'") ? t.slice(0, -1) : t.endsWith("2") ? t : t + "'"))
    .join(" ");
}

/** 같은 면을 연달아 돌린 것을 합친다 (U U → U2, U U' → 없음) */
export function simplify<T extends ColorMove>(ms: readonly T[]): T[] {
  const out: T[] = [];
  for (const m of ms) {
    const last = out[out.length - 1];
    if (last && last.color === m.color) {
      const k = (last.k + m.k) % 4;
      out.pop();
      if (k) out.push({ ...last, k: k as 1 | 2 | 3 });
    } else out.push({ ...m });
  }
  return out;
}

// --- 조각 ------------------------------------------------------------------

/** 조각 하나 = 그 조각에 붙은 칸들. 모서리는 2칸, 꼭짓점은 3칸. */
function pieces(n: number): number[][] {
  const byPos = new Map<string, number[]>();
  FACELET_POS.forEach((p, i) => {
    const key = p.join(",");
    byPos.set(key, [...(byPos.get(key) ?? []), i]);
  });
  return [...byPos.values()].filter((v) => v.length === n);
}

export const EDGES = pieces(2);
export const CORNERS = pieces(3);

/** 색 집합으로 조각을 찾는다: 지금 그 색 조합의 조각이 어느 칸들에 있는지 */
export function findPiece(s: State, colors: readonly Color[]): number[] {
  const set = [...colors].sort().join();
  for (const p of colors.length === 2 ? EDGES : CORNERS) {
    if (p.map((i) => s[i]).sort().join() === set) return p;
  }
  throw new Error("그런 조각이 없음");
}

/** 조각이 자리에 맞는가: 각 칸의 색이 그 칸이 속한 면의 가운데 색과 같다 */
export function pieceSolved(s: State, cells: readonly number[]): boolean {
  return cells.every((i) => s[i] === s[Math.floor(i / 9) * 9 + 4]);
}

/** 색 c 스티커가 지금 있는 칸 (조각 안에서) */
export function cellOfColor(s: State, piece: readonly number[], c: Color): number {
  return piece.find((i) => s[i] === c)!;
}

/** 칸 i가 속한 면의 가운데 색 */
export const faceColorOfCell = (s: State, i: number): Color => s[Math.floor(i / 9) * 9 + 4];

export function isSolved(s: State): boolean {
  for (let f = 0; f < 6; f++) for (let j = 0; j < 9; j++) if (s[f * 9 + j] !== s[f * 9 + 4]) return false;
  return true;
}

export function randomScramble(n: number, rand: () => number = Math.random): string {
  const out: string[] = [];
  while (out.length < n) {
    const m = FACES[Math.floor(rand() * 6)] + ["", "2", "'"][Math.floor(rand() * 3)];
    if (out.length && out[out.length - 1][0] === m[0]) continue;
    out.push(m);
  }
  return out.join(" ");
}

/** 재현 가능한 난수 (테스트용) */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 큐브 그림 (SVG). 사람이 지금 들고 있는 방향 그대로, 위·앞·오른쪽 세 면을 입체로,
 * 안 보이는 세 면(왼쪽·뒤·아래)은 작은 평면으로 그린다.
 * 다음에 돌릴 층은 테두리로 표시하고, 돌아가는 중인 층은 그 각도만큼 기울여 그린다.
 */

import type { State, Vec3, View } from "../coach/cube.ts";
import { FACELET_NORMAL, FACELET_POS, SLOT_NORMAL, slotOfColor } from "../coach/cube.ts";

export const HEX = ["#f2f2ee", "#d7263d", "#13985a", "#ffd23f", "#ff7a1a", "#1d5fd3"];

type V = [number, number, number];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a: V, b: V, k = 1): V => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];

/** 큐브 좌표 → 보는 사람 좌표 (x 오른쪽, y 위, z 자기 쪽) */
function basis(s: State, v: View): (p: Vec3) => V {
  const u = SLOT_NORMAL[slotOfColor(s, v.up)];
  const f = SLOT_NORMAL[slotOfColor(s, v.front)];
  const r = cross(u, f);
  return (p) => [dot(p, r), dot(p, u), dot(p, f)];
}

/** 축 a 둘레로 각도 t만큼 회전 (바깥에서 볼 때 시계 방향이 +) */
function rot(p: V, axis: V, t: number): V {
  const c = Math.cos(-t);
  const s = Math.sin(-t);
  const k = axis;
  const d = dot(k, p);
  const x = cross(k, p);
  return [p[0] * c + x[0] * s + k[0] * d * (1 - c), p[1] * c + x[1] * s + k[1] * d * (1 - c), p[2] * c + x[2] * s + k[2] * d * (1 - c)];
}

const iso = (p: V): [number, number] => [(p[0] - p[2]) * 0.866, -p[1] + (p[0] + p[2]) * 0.5];

export interface Highlight {
  /** 보는 사람 좌표에서 돌릴 층의 바깥 방향 (예: 오른쪽 면이면 [1,0,0]) */
  axis: V;
  /** 돌아가는 중이면 그 각도 (라디안) */
  angle?: number;
  /** 다음에 돌릴 층(안내) · 되돌려야 하는 층(바로잡기) · 지금 돌아가는 층 */
  mode: "next" | "fix" | "turning";
}

export function renderCube(s: State, v: View, hl: Highlight | null, size = 300): string {
  const toV = basis(s, v);
  const polys: { depth: number; svg: string }[] = [];
  const scale = size / 6.2;
  const cx = size / 2;
  const cy = size / 2;
  for (let i = 0; i < 54; i++) {
    let n = toV(FACELET_NORMAL[i]);
    let c = add(toV(FACELET_POS[i]), n, 0.5);
    const inLayer = hl ? Math.round(dot(toV(FACELET_POS[i]), hl.axis)) === 1 : false;
    if (inLayer && hl?.angle) {
      c = rot(c, hl.axis, hl.angle);
      n = rot(n, hl.axis, hl.angle);
    }
    // 보는 방향 (1,1,1) 쪽을 향한 면만
    if (dot(n, [1, 1, 1]) <= 0.05) continue;
    const a: V = Math.abs(n[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
    const a2 = cross(n, a);
    const a1 = cross(a2, n);
    const k = 0.44;
    const pts = [
      [k, k],
      [k, -k],
      [-k, -k],
      [-k, k],
    ].map(([p, q]) => iso(add(add(c, a1, p), a2, q)));
    const body = [
      [0.5, 0.5],
      [0.5, -0.5],
      [-0.5, -0.5],
      [-0.5, 0.5],
    ].map(([p, q]) => iso(add(add(c, a1, p), a2, q)));
    const P = (q: [number, number][]) => q.map(([x, y]) => `${(cx + x * scale).toFixed(1)},${(cy + y * scale).toFixed(1)}`).join(" ");
    const stroke = inLayer ? (hl!.mode === "turning" ? "var(--turn)" : hl!.mode === "fix" ? "var(--fix)" : "var(--next)") : "none";
    polys.push({
      depth: dot(c, [1, 1, 1]),
      svg: `<polygon points="${P(body)}" fill="#151515"/><polygon points="${P(pts)}" fill="${HEX[s[i]]}" stroke="${stroke}" stroke-width="${inLayer ? 3 : 0}" stroke-linejoin="round"/>`,
    });
  }
  polys.sort((x, y) => x.depth - y.depth);
  return `<svg viewBox="0 0 ${size} ${size}" width="100%" role="img" aria-label="큐브">${polys.map((p) => p.svg).join("")}</svg>`;
}

/** 방향 그림에 칠할 세 면의 색 (사람 시점의 위·앞·오른쪽 가운데 색) */
export interface FaceColors {
  up: number;
  front: number;
  right: number;
}

/**
 * 다음 동작 방향 그림: 돌릴 면 둘레를 도는 화살표.
 * 큐브 그림과 같은 투영·같은 회전 규칙(rot)을 써서, 화살표가 큐브 애니메이션과 어긋나지 않게 한다.
 * axis는 보는 사람 좌표에서 그 면의 바깥 방향, k는 바깥에서 봐서 시계 방향 횟수.
 * faces를 주면 세 면을 실제 가운데 색으로 칠한다 (손에 든 큐브와 맞춰 보라고).
 */
export function renderMoveArrow(axis: V, k: 1 | 2 | 3, wide = false, size = 84, faces?: FaceColors): string {
  const scale = size / 7.4;
  const cx = size / 2;
  const cy = size / 2;
  const pt = (p: V): [number, number] => {
    const [x, y] = iso(p);
    return [cx + x * scale, cy + y * scale];
  };
  const str = (q: [number, number][]) => q.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  // 큐브 윤곽 (보이는 세 면). 돌릴 면이 뒤쪽이면 전체를 흐리게 해서 "반대쪽"임을 보인다
  const back = dot(axis, [1, 1, 1]) <= 0.05;
  const fill = faces ? [faces.up, faces.front, faces.right].map((c) => HEX[c]) : null;
  const body = ([[0, 1, 0], [0, 0, 1], [1, 0, 0]] as V[])
    .map((n, fi) => {
      const a: V = Math.abs(n[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      const a2 = cross(n, a);
      const a1 = cross(a2, n);
      const c = add([0, 0, 0], n, 1.5);
      const at = (p: number, q: number) => pt(add(add(c, a1, p), a2, q));
      const quad = [at(1.5, 1.5), at(1.5, -1.5), at(-1.5, -1.5), at(-1.5, 1.5)];
      const on = !back && dot(n, axis) > 0.9;
      if (!fill) {
        return `<polygon points="${str(quad)}" fill="${on ? "currentColor" : "var(--arrow-face)"}" fill-opacity="${on ? 0.45 : 1}" stroke="var(--arrow-line)" stroke-width="1" stroke-linejoin="round"/>`;
      }
      // 칸 선 (3x3)
      const grid = [-0.5, 0.5]
        .map((t) => `<polyline points="${str([at(t, -1.5), at(t, 1.5)])}"/><polyline points="${str([at(-1.5, t), at(1.5, t)])}"/>`)
        .join("");
      return (
        `<polygon points="${str(quad)}" fill="${fill[fi]}" fill-opacity="${on || back ? 1 : 0.78}"/>` +
        `<g stroke="#151515" stroke-opacity="0.45" stroke-width="${size / 90}" fill="none">${grid}</g>` +
        `<polygon points="${str(quad)}" fill="none" stroke="${on ? "currentColor" : "#151515"}" stroke-opacity="${on ? 1 : 0.6}" stroke-width="${on ? size / 30 : size / 80}" stroke-linejoin="round"/>`
      );
    })
    .join("");

  // 면 바깥을 도는 화살표
  const perp: V = Math.abs(axis[1]) > 0.9 ? [0, 0, 1] : [0, 1, 0];
  const e1 = cross(axis, perp);
  const total = (k === 2 ? Math.PI : Math.PI / 2) * (k === 3 ? -1 : 1);
  const arc = (radius: number, lift: number): string => {
    const start = add(add([0, 0, 0], axis, lift), e1, radius);
    const pts: [number, number][] = [];
    for (let i = 0; i <= 18; i++) pts.push(pt(rot(start, axis, (total * i) / 18)));
    const [hx, hy] = pts[pts.length - 1];
    const [bx, by] = pts[pts.length - 2];
    const dx = hx - bx;
    const dy = hy - by;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const h = size / 9;
    const head: [number, number][] = [
      [hx + ux * h, hy + uy * h],
      [hx - ux * h * 0.3 - uy * h * 0.62, hy - uy * h * 0.3 + ux * h * 0.62],
      [hx - ux * h * 0.3 + uy * h * 0.62, hy - uy * h * 0.3 - ux * h * 0.62],
    ];
    const w = size / 24;
    // 색 칠한 면 위에서도 보이게 바탕색 테두리를 먼저 깐다
    const halo = fill
      ? `<polyline points="${str(pts)}" fill="none" stroke="var(--panel)" stroke-width="${w + size / 28}" stroke-linecap="round"/><polygon points="${str(head)}" fill="var(--panel)" stroke="var(--panel)" stroke-width="${size / 28}" stroke-linejoin="round"/>`
      : "";
    return `${halo}<polyline points="${str(pts)}" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round"/><polygon points="${str(head)}" fill="currentColor"/>`;
  };
  const arrows = wide ? arc(1.75, 2.1) + arc(1.75, 0.9) : arc(1.75, 2.1);
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img" aria-label="돌릴 방향"><g opacity="${back ? (fill ? 0.6 : 0.55) : 1}">${body}</g>${arrows}</svg>`;
}

/** 안 보이는 면: 그 면을 바깥에서 바라본 모습 */
export function renderHidden(s: State, v: View): string {
  const toV = basis(s, v);
  const faces: { name: string; n: V; right: V; up: V }[] = [
    { name: "왼쪽", n: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] },
    { name: "뒤", n: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] },
    { name: "아래", n: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] },
  ];
  return faces
    .map((f) => {
      const cells: string[] = new Array(9).fill("");
      for (let i = 0; i < 54; i++) {
        const n = toV(FACELET_NORMAL[i]);
        if (dot(n, f.n) < 0.9) continue;
        const p = toV(FACELET_POS[i]);
        const col = Math.round(dot(p, f.right)) + 1;
        const row = 1 - Math.round(dot(p, f.up));
        cells[row * 3 + col] = HEX[s[i]];
      }
      const rects = cells.map((col, j) => `<rect x="${(j % 3) * 11 + 1}" y="${Math.floor(j / 3) * 11 + 1}" width="10" height="10" rx="1.5" fill="${col}"/>`).join("");
      return `<figure class="mini"><svg viewBox="0 0 34 34" width="56" height="56"><rect width="34" height="34" rx="3" fill="#151515"/>${rects}</svg><figcaption>${f.name}</figcaption></figure>`;
    })
    .join("");
}

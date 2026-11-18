/**
 * 브라우저용 연속 추적기. 파이썬 src/cubecoach/tracker.py 를 그대로 옮겼다
 * (같은 입력에 같은 점수를 내는지 test/tracker.test.ts 가 파이썬 정답값과 대조한다).
 *
 * 입력: 인식 모델이 관심영역(192x192 → 48x48 지도)마다 내는
 *   logp (8, n, n): 칸마다 [스티커 색 6 / 큐브 몸체 / 그 밖]의 로그확률
 *   heat (n, n)   : 스티커 중심일 가능성 0~1
 * 추적하는 것: 큐브 몸통의 자세 R·T, 지금 돌아가는 층과 그 각도.
 * 내는 것: 확정한 회전(가운데 색, 시계 방향 90도 단위), 도는 중인 층과 각도, 보는 방향.
 *
 * 파이썬판과 다른 것: 90도에 닿아 확정할 때 '회전 없음'과도 견준다 (snapVsNone, 가짜 회전 줄이기).
 * 파이썬판에 더한 것 하나: 색을 모르는 칸(UNKNOWN). 처음 상태를 읽을 때(스캔)는 가운데 칸만
 * 알고 나머지는 "스티커가 있다"는 것만 맞춰 자세를 잡은 뒤, 보이는 칸의 색을 모은다.
 */

import T from "./tables.json" with { type: "json" };

export const UNKNOWN = 255;
const DEG = Math.PI / 180;
const BODY = 6;
const OTHER = 7;
const NC = 8;

const CENTERS = Float64Array.from(T.centers.flat());
const NORMALS = Float64Array.from(T.normals.flat());
const AXES = Float64Array.from(T.axes.flat());
const LAYER = Uint8Array.from(T.layer.flat());
export const CAND_PERMS: Int16Array[] = T.candPerms.map((p) => Int16Array.from(p));
export const CAND_EVENTS = T.candEvents as [number, number][];

// 가설 → 화면
const P0 = 0.2;
const WRONG = -3.0, ON_BODY = -1.2, OCCLUDED = -0.4, OUTSIDE = -1.0;
const MATCH_MAX = 2.0;
const HEAT_BONUS = 1.5;
// 화면 → 가설
const PEAK_MIN = 0.35, PEAK_MAX_N = 48;
const UNEXPLAINED = 2.0;
const PEAK_SIGMA = 1.3;

export const ROI_SCALE = 2.8;
const INERTIA_ROT = 1.0 / 10 ** 2, INERTIA_T = 3.0, INERTIA_S = 0.5 / 0.05 ** 2;
const TURN_START_GAIN = 3.0;
const SNAP = 14 * DEG, CANCEL = 8 * DEG;
const DOUBT_QUALITY = 0.92, DOUBT_FRAMES = 3;
const RECOVER_MARGIN = 5.0, RECOVER_LEAD = 1.0, TIE = 2.0;
const HISTORY = 3;
const LOST_QUALITY = 0.45, LOST_FRAMES = 10;
// 실제 영상에서 더한 안전장치: 스티커로 읽힌 칸 가운데 지금 가설이 설명하는 비율(덮는 비율).
// 큐브 크기를 작게 잘못 잡으면 다음 관심영역이 좁아져 큐브 일부만 보이고, 그 일부의 색은 맞아서
// 일치율(quality)은 높게 나온다. 덮는 비율은 이런 상태를 잡아낸다. 낮은 채로 이어지면 상태는 두고 자세만 넓게 다시 찾는다.
export const COVER_MIN = 0.6, COVER_FRAMES = 3;
// 자세를 새로 잡은 직후(스캔 직후, 놓친 뒤 다시 찾은 직후)와 큐브를 통째로 빠르게 돌리는 동안에는
// 층 회전을 찾지 않는다. 실제 영상에서 가짜 회전은 대부분 이때 나왔다(고쳐 잡기·면 보여 주기).
// 일치율이 높고 몸통이 거의 안 도는 프레임이 SETTLE_FRAMES만큼 이어지면 다시 찾는다.
export const SETTLE_QUALITY = 0.8, FAST_BODY_DEG = 12;
// 큐브를 옮길 때 빨리 따라가기 (follow). 문턱은 실제 영상에서 가만히 돌리는 구간(B, 시드 5개)의
// 프레임 분포로 정했다: 그 구간에서는 한 번도 켜지지 않아 예전 흐름과 결과가 똑같다.
//  - 자세 중심 속도(눌러 잰 값): B 최대 0.315 조각/프레임, 빠르게 옮길 때 0.5~1
//  - 큐브 영역 무게중심과 자세 중심의 어긋남: B 최대 1.5 조각
//  - 일치율이 정확히 0(자세가 큐브 위에 아예 없음)인 프레임: B에는 없음
export const MOVE_ON = 0.35; // 이보다 빠르면 '옮기는 중': 한 프레임 앞을 내다보고 넓게 흔든다
const VEL_SMOOTH = 0.5;
const VEL_MAX = 1.5; // 한 프레임에 내다보는 최대 거리 (조각)
const MOVE_ROI_GROW = 1.15; // 옮기는 중에는 관심영역을 조금 넓힌다 (예측이 빗나가도 큐브가 들어오게)
const WIDE_ROUND: [number, number, number, number] = [9.0, 0.45, 0.05, 6.0]; // 옮기는 중에만 앞에 더하는 넓은 흔들기
export const JUMP_OFF = 1.7; // 무게중심이 이만큼 넘게 어긋났을 때만 그쪽으로 옮겨 본다
const JUMP_GAIN = 2.0; // 옮긴 자세가 지금보다 이만큼은 더 잘 맞아야 옮긴다
export const ZERO_LOST = 2; // 일치율 0이 이만큼 이어지면 10프레임을 기다리지 않고 바로 다시 찾는다
const RELOCK_ROT_SIG = 22, RELOCK_N = 48, RELOCK_SCALES = [0.85, 1.0, 1.25];
const RELOCK_QUALITY = 0.75; // 둘레에서 다시 잡은 자세는 이만큼 잘 맞아야 받아들인다 (아니면 전체 탐색)
const CALM_AFTER = 3; // 옮기기를 멈춘 뒤 이만큼은 층 회전을 새로 찾지 않는다
const SEEK_TRIES = 2; // 넓게 찾다 본 큐브 쪽으로 좁혀 찾기를 몇 번까지

export type Roi = [number, number, number]; // 중심 x, 중심 y, 한 변 (프레임 너비 기준, 프레임 중심이 0)

/** 큐브를 옮길 때 빨리 따라가기의 세부 스위치 (하나씩 켜서 효과를 잴 수 있게 나눴다) */
export interface FollowOpts {
  /** 등속 예측: 관심영역과 출발 자세를 한 프레임 앞으로 */
  predict: boolean;
  /** 분할 지도의 큐브 영역 무게중심으로 출발 자세 옮기기 */
  jump: boolean;
  /** 놓친 직후 전체 탐색 전에 직전 자세 둘레부터 다시 찾기 */
  relock: boolean;
  /** 놓쳤다는 판정을 빨리 */
  fastLost: boolean;
  /** 놓쳤을 때 화면 전체 너비도 번갈아 찾고, 본 쪽으로 좁혀 다시 찾기 */
  wide: boolean;
  /** 다시 잡은 직후 몇 프레임은 층 회전을 새로 찾지 않기 (다시 잡은 자세가 덜 맞을 때 가짜 회전 방지) */
  calmAfterLock: boolean;
}
const FOLLOW_OFF: FollowOpts = { predict: false, jump: false, relock: false, fastLost: false, wide: false, calmAfterLock: false };

export interface Maps {
  n: number;
  logp: Float32Array; // NC * n * n
  heat: Float32Array; // n * n
}

export interface Event {
  center: number; // 돌린 층의 가운데 색
  quarterTurns: number; // 시계 방향 90도 단위
  frame: number;
  /** 어떻게 확정했나 (가짜 회전 분석용 진단값) */
  info?: EventInfo;
}

/** 회전 하나가 어떻게 판정됐는지 (진단용, 판정에는 쓰지 않는다) */
export interface EventInfo {
  /** snap: 90도에 닿아 확정 / snapOther: 닿았지만 다른 회전으로 확정 / cancel: 돌다 0도로 멈춘 뒤 / doubt: 일치율이 낮아 다시 견줘서 */
  via: "snap" | "snapOther" | "cancel" | "doubt" | "revert";
  /** 몇 수 앞에 끼워 넣었나 (0 = 지금) */
  back: number;
  /** 회전 시작 가설: 층을 잡고 몸통이 돈 경우(true) / 층이 돈 경우(false) / 시작 없이(null) */
  held: boolean | null;
  /** 시작부터 확정까지 프레임 */
  turnFrames: number;
  /** 직전 일치율 */
  q: number;
  /** 최근 3프레임 몸통 회전 속도 (도/프레임) */
  body: number;
  /** 시작할 때 층 회전 가설이 기존 자세보다 나은 점수 */
  gain: number;
  /** 같은 때 통째 회전 가설(층 없이 몸통만 15~75도)이 기존 자세보다 나은 점수 */
  bodyGain: number;
  /** 확정한 상태가 '회전 없음'보다 나은 점수 */
  vsNone: number;
  /** 돌린 층의 21칸 가운데 보이는 정도의 합 (확정 자세 기준) */
  seen: number;
}

interface Hyps {
  n: number;
  R: Float64Array; // n * 9 (행 우선)
  T: Float64Array; // n * 3
  layer: Int32Array;
  phi: Float64Array;
}

// --- 작은 선형대수 -----------------------------------------------------------

/** 회전 벡터 → 회전 행렬 (Rodrigues), out[o..o+9] */
export function rotvec(wx: number, wy: number, wz: number, out: Float64Array = new Float64Array(9), o = 0): Float64Array {
  const th = Math.hypot(wx, wy, wz);
  if (th < 1e-12) {
    out.fill(0, o, o + 9);
    out[o] = out[o + 4] = out[o + 8] = 1;
    return out;
  }
  const x = wx / th, y = wy / th, z = wz / th;
  const s = Math.sin(th), c = 1 - Math.cos(th);
  out[o] = 1 + c * (-y * y - z * z);
  out[o + 1] = -s * z + c * x * y;
  out[o + 2] = s * y + c * x * z;
  out[o + 3] = s * z + c * x * y;
  out[o + 4] = 1 + c * (-x * x - z * z);
  out[o + 5] = -s * x + c * y * z;
  out[o + 6] = -s * y + c * x * z;
  out[o + 7] = s * x + c * y * z;
  out[o + 8] = 1 + c * (-x * x - y * y);
  return out;
}

/** out = a @ b (3x3) */
function mul3(a: Float64Array, ao: number, b: Float64Array, bo: number, out: Float64Array, oo: number): void {
  const t = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) t[i * 3 + j] = a[ao + i * 3] * b[bo + j] + a[ao + i * 3 + 1] * b[bo + 3 + j] + a[ao + i * 3 + 2] * b[bo + 6 + j];
  out.set(t, oo);
}

/** 균등 분포 무작위 회전 n개 */
export function randomRotations(rand: () => number, n: number): Float64Array {
  const out = new Float64Array(n * 9);
  for (let i = 0; i < n; i++) {
    let w = gauss(rand), x = gauss(rand), y = gauss(rand), z = gauss(rand);
    const q = Math.hypot(w, x, y, z);
    w /= q; x /= q; y /= q; z /= q;
    out.set([
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], i * 9);
  }
  return out;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rand: () => number): number {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function makeHyps(n: number): Hyps {
  return { n, R: new Float64Array(n * 9), T: new Float64Array(n * 3), layer: new Int32Array(n).fill(-1), phi: new Float64Array(n) };
}

const permute = (s: Uint8Array, p: Int16Array): Uint8Array => {
  const o = new Uint8Array(54);
  for (let i = 0; i < 54; i++) o[i] = s[p[i]];
  return o;
};

// --- 추적기 ------------------------------------------------------------------

/** 한 프레임 지도에서 미리 계산해 두는 것: 칸마다 가장 그럴듯한 분류, 스티커일 로그확률 */
interface Prepared {
  maps: Maps;
  cls: Uint8Array;
  stickerLp: Float32Array; // log(스티커 색 6개 확률의 합)
  peaks: { xy: Float64Array; col: Uint8Array; w: Float64Array; n: number };
}

export function prepare(maps: Maps): Prepared {
  const { n, logp, heat } = maps;
  const nn = n * n;
  const cls = new Uint8Array(nn);
  const stickerLp = new Float32Array(nn);
  for (let p = 0; p < nn; p++) {
    let best = 0, bv = logp[p], sum = 0;
    for (let c = 0; c < NC; c++) {
      const v = logp[c * nn + p];
      if (v > bv) { bv = v; best = c; }
      if (c < 6) sum += Math.exp(v);
    }
    cls[p] = best;
    stickerLp[p] = Math.log(Math.max(sum, 1e-12));
  }
  // 중심 지도의 봉우리 = 화면에 보이는 스티커
  const cand: { y: number; x: number; h: number }[] = [];
  const H = (y: number, x: number) => (y < 0 || x < 0 || y >= n || x >= n ? 0 : heat[y * n + x]);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const h = heat[y * n + x];
      if (h <= PEAK_MIN) continue;
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) for (let dx = -1; dx <= 1; dx++) if (H(y + dy, x + dx) > h) { isMax = false; break; }
      if (!isMax || Math.exp(stickerLp[y * n + x]) <= 0.5) continue;
      cand.push({ y, x, h });
    }
  // 파이썬(np.nonzero 순서 = 행 우선)과 같은 순서를 유지하되, 너무 많으면 센 것만
  let keep = cand;
  if (cand.length > PEAK_MAX_N) {
    const thr = [...cand].sort((a, b) => b.h - a.h)[PEAK_MAX_N - 1].h;
    keep = cand.filter((c) => c.h >= thr).slice(0, PEAK_MAX_N);
  }
  const m = keep.length;
  const xy = new Float64Array(m * 2), col = new Uint8Array(m), w = new Float64Array(m);
  keep.forEach((c, i) => {
    let sw = 0, ox = 0, oy = 0;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const v = H(c.y + dy, c.x + dx);
        sw += v; ox += v * dx; oy += v * dy;
      }
    xy[i * 2] = c.x + 0.5 + ox / Math.max(sw, 1e-6);
    xy[i * 2 + 1] = c.y + 0.5 + oy / Math.max(sw, 1e-6);
    let best = 0;
    for (let k = 1; k < 6; k++) if (logp[k * nn + c.y * n + c.x] > logp[best * nn + c.y * n + c.x]) best = k;
    col[i] = best;
    w[i] = c.h;
  });
  return { maps, cls, stickerLp, peaks: { xy, col, w, n: m } };
}

export class Tracker {
  colors: Uint8Array;
  f: number;
  R: Float64Array | null = null;
  Tv: Float64Array | null = null;
  layer = -1;
  phi = 0;
  frame = 0;
  quality = 0;
  events: Event[] = [];
  /** 스캔하는 동안은 층 회전을 찾지 않는다 (색을 모르는 칸은 돌려도 구별되지 않으므로) */
  allowTurns = true;
  /**
   * 덮는 비율이 낮으면 자세를 넓게 다시 찾기. 실제 영상 4구간 x 시드 5개 비교에서 효과가 없어(README 9절 사례 4)
   * 기본은 끈다 (끄면 파이썬판과 같은 흐름).
   */
  useCover = false;
  /**
   * 자세를 새로 잡은 뒤 회전 판정을 쉬는 안정 프레임 수 (0이면 끔). 같은 비교에서 진짜 회전까지 막아
   * 정확도가 크게 떨어졌으므로(느린 구간 0.84 → 0.15) 기본은 끈다.
   */
  settleFrames = 0;
  /**
   * 큐브를 옮길 때 빨리 따라가기 (기본 끔 = 예전 흐름, 앱은 camera.ts 에서 켠다). 세부는 FollowOpts.
   * 문턱을 넘을 때만 작동하므로 가만히 돌리는 동안에는 예전과 똑같이 움직인다.
   * fastFollow = true/false 로 한꺼번에 켜고 끈다.
   */
  follow: FollowOpts = { ...FOLLOW_OFF };
  get fastFollow(): boolean {
    return Object.values(this.follow).some(Boolean);
  }
  set fastFollow(on: boolean) {
    for (const k of Object.keys(this.follow) as (keyof FollowOpts)[]) this.follow[k] = on;
  }
  /**
   * 층 회전을 시작하려면 같은 때 '큐브가 통째로 돌았다'(층 고정 없이 몸통만) 가설보다 이만큼은 더 잘 맞아야 한다 (0이면 끔).
   * 빠르게 통째로 돌리면 흔들기(9도)가 못 따라가 층 회전 가설이 이기는 일이 많았다.
   */
  turnVsBody = 0;
  /** 층 회전을 시작하려면 기존 자세보다 이만큼 더 잘 맞아야 한다 (파이썬판과 같은 기본 3) */
  turnStartGain = TURN_START_GAIN;
  /** 위 조건으로 층 회전을 버릴 때 통째 회전 가설이 기존 자세보다 낫으면 그 자세로 옮긴다 */
  adoptBody = false;
  /**
   * 90도에 닿아 확정할 때 고른 상태가 '회전 없음'보다 이만큼은 나아야 한다 (-Infinity면 끔 = 파이썬판 규칙).
   * 파이썬판 규칙은 추적한 회전이 '회전 없음'보다 2점 나빠도 받아들이고, 다른 회전으로 바꿀 때는 '회전 없음'과
   * 견주지 않았다. 실제 영상의 회전 없는 구간(들고 있기·고쳐 잡기)에서 나온 가짜 회전 대부분이 이 길로 들어왔다.
   * 값은 회전 없는 구간의 가짜 회전 수와 B 구간 정확도를 함께 보고 골랐다 (reports/track_real_falseturn_summary.json).
   */
  snapVsNone = 4;
  /** 90도에 닿았는데 추적한 층과 다른 회전을 고를 때는 '회전 없음'보다 이만큼 나아야 한다 */
  snapOtherVsNone = 8;
  /**
   * 직전 회전 확정 뒤 이 프레임 수 안이면(빠르게 이어 돌리는 중) 위 두 조건을 걸지 않는다. 기본 0 = 늘 건다.
   * 처음에는 20으로 두었다. 늘 엄격하면 빠른 섹시무브(A 구간)가 0.21로 무너지는 것처럼 보였기 때문인데,
   * 이는 정답 오류였다(덧붙은 한 수가 실제로는 13초에 큐브를 떨어뜨릴 때 돌아갔다). 정답을 고치고 다시 재니
   * 늘 엄격 0.36 vs 20프레임 예외 0.38로 비슷했고, 천천히 돌린 B 구간은 0.97 vs 0.84로 늘 엄격이 나았다
   * (reports/track_real_gt2_summary.json). 프레임 수라 fps에 따라 길이가 바뀌는 문제도 없앤다.
   */
  relaxWithin = 0;
  private lastMoveFrame = -1e9;
  /** 위 조건들로 버린 층 회전 수 (진단용) */
  rejectedTurns = 0;
  /**
   * 확정 뒤 검증: 이 프레임 수 동안 '돌리기 전 상태'가 지금 상태보다 revertMargin 넘게 더 잘 맞는 프레임이
   * revertWins 번 나오면 그 회전을 되돌린다 (되돌림은 반대 회전 이벤트로 낸다). 0이면 끔.
   * 의심 판정(decide)의 후보에는 '직전 회전 취소'가 없어서, 가짜 회전이 한 번 들어가면 상태가 계속 틀어졌다.
   */
  revertFrames = 0;
  revertMargin = 3;
  revertWins = 2;
  /** 되돌린 회전 수 (진단용) */
  reverts = 0;
  private rand2: () => number;
  private verify: { left: number; wins: number; before: Uint8Array; event: Event; frame: number } | null = null;
  private bodyBank: Float64Array;
  /** 프레임마다 남기는 진단값: 큐브 영역 무게중심과 자세 중심의 어긋남·자세 중심 속도 (조각 단위) */
  diag = { blobOff: -1, speed: 0, jumpGain: 0, bodyDeg: 0 };
  private settle = 0;
  private rand: () => number;
  private bank: Float64Array;
  private still = 0;
  private idle = 0;
  private badFrames = 0;
  private lowCover = 0;
  coverage = 1;
  /** 자세를 넓게 다시 찾은 횟수 (진단용) */
  relocks = 0;
  private doubt = 0;
  private log: { before: Uint8Array; perm: Int16Array }[] = [];
  private prep: Prepared | null = null;
  // follow 용
  private prevT: Float64Array | null = null;
  private vel = new Float64Array(3);
  private lastR: Float64Array | null = null;
  private lastT: Float64Array | null = null;
  private zeroRun = 0;
  private lostFrames = 0;
  private seek: { u: number; v: number; side: number; tries: number } | null = null;
  /** 지금 '옮기는 중'으로 보고 있나 (진단용) */
  moving = false;
  private calm = 0;
  // 진단용 (가짜 회전 분석)
  private turnStart: { held: boolean; frame: number; gain: number; bodyGain: number } | null = null;
  private prevRend: Float64Array | null = null;
  private bodyHist: number[] = [];
  /** 전체 탐색을 한 횟수 (진단용) */
  inits = 0;
  /** 전체 탐색 대신 직전 자세 둘레에서 다시 잡은 횟수 (진단용) */
  localRelocks = 0;

  constructor(colors: Uint8Array, f = 0.95, seed = 0, bankSize = 2500) {
    this.colors = Uint8Array.from(colors);
    this.f = f;
    this.rand = mulberry32(seed);
    this.rand2 = mulberry32(seed + 1000); // 검증용 (추적 난수 흐름을 바꾸지 않게 따로)
    this.bank = randomRotations(mulberry32(123), bankSize);
    // 통째 회전 가설: 세 축 둘레 ±15~75도 30개 + 무작위 축 둘레 10~60도 40개 (고정 난수라 추적 난수와 따로 논다)
    const r = mulberry32(77), D = 3 * 10 + 40;
    this.bodyBank = new Float64Array(D * 9);
    const angles = [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75];
    for (let i = 0; i < D; i++) {
      if (i < 30) {
        const L = Math.floor(i / 10), a = angles[i % 10] * DEG;
        rotvec(a * AXES[L * 3], a * AXES[L * 3 + 1], a * AXES[L * 3 + 2], this.bodyBank, i * 9);
      } else {
        let x = gauss(r), y = gauss(r), z = gauss(r);
        const n = Math.hypot(x, y, z), a = (10 + 50 * r()) * DEG;
        x /= n; y /= n; z /= n;
        rotvec(a * x, a * y, a * z, this.bodyBank, i * 9);
      }
    }
  }

  /**
   * 다음 프레임에서 인식 모델에 넣을 정사각 영역 (프레임 너비 기준).
   * 큐브를 옮기는 중이면(follow.predict) 한 프레임 앞을 내다본 자리에 조금 넓게 잡는다.
   */
  roi(): Roi | null {
    if (!this.Tv) return null;
    let [x, y] = this.Tv;
    const z = this.Tv[2];
    let grow = 1;
    if (this.isMoving()) {
      x += this.vel[0];
      y += this.vel[1];
      grow = MOVE_ROI_GROW;
    }
    const unit = this.f / z;
    return [(this.f * x) / z, (this.f * y) / z, 3 * unit * ROI_SCALE * grow];
  }

  /** 옮기는 중인가: 눌러 잰 자세 중심 속도가 문턱보다 큰가 (follow.predict 일 때만) */
  private isMoving(): boolean {
    return this.follow.predict && this.prevT !== null && Math.hypot(this.vel[0], this.vel[1]) > MOVE_ON;
  }

  /**
   * 놓쳤을 때 큐브를 찾을 영역. 기본은 화면 가운데의 높이만 한 정사각형.
   * follow.wide 면 화면 전체 너비와 번갈아 찾고, 넓게 찾다 큐브로 보이는 것을 봤으면 그쪽으로 좁혀 찾는다
   * (화면 가장자리로 옮긴 큐브도 찾고, 넓게 볼 때의 낮은 해상도로 자세를 맞추지 않게).
   */
  lostRoi(aspect: number): Roi {
    if (!this.follow.wide) return [0, 0, aspect];
    if (this.seek) return [this.seek.u, this.seek.v, Math.max(this.seek.side * ROI_SCALE, 0.25)];
    return this.lostFrames % 2 === 1 ? [0, 0, 1] : [0, 0, aspect];
  }

  reset(): void {
    if (this.R && this.Tv) {
      this.lastR = this.R.slice(); // 다시 찾을 때 둘레부터 훑으려고 기억해 둔다
      this.lastT = this.Tv.slice();
    }
    this.R = this.Tv = null;
    this.prevT = null;
    this.diagPrev = null;
    this.vel.fill(0);
    this.moving = false;
    this.calm = 0;
    this.turnStart = null;
    this.verify = null;
    this.prevRend = null;
    this.bodyHist = [];
    this.zeroRun = this.lostFrames = 0;
    this.seek = null;
    this.layer = -1;
    this.phi = 0;
    this.badFrames = this.doubt = this.still = this.idle = this.lowCover = 0;
  }

  // --- 투영과 점수 ------------------------------------------------------------

  /** 가설마다 54칸 중심의 지도 좌표 (n*54*2)와 신뢰 가중치 (n*54) */
  project(h: Hyps, roi: Roi, n: number): { px: Float64Array; w: Float64Array } {
    const px = new Float64Array(h.n * 108), w = new Float64Array(h.n * 54);
    const M = new Float64Array(9);
    const f = this.f;
    for (let i = 0; i < h.n; i++) {
      const L = h.layer[i];
      if (L >= 0) rotvec(-h.phi[i] * AXES[L * 3], -h.phi[i] * AXES[L * 3 + 1], -h.phi[i] * AXES[L * 3 + 2], M);
      const r = h.R, ro = i * 9, tx = h.T[i * 3], ty = h.T[i * 3 + 1], tz = h.T[i * 3 + 2];
      for (let k = 0; k < 54; k++) {
        let cx = CENTERS[k * 3], cy = CENTERS[k * 3 + 1], cz = CENTERS[k * 3 + 2];
        let nx = NORMALS[k * 3], ny = NORMALS[k * 3 + 1], nz = NORMALS[k * 3 + 2];
        if (L >= 0 && LAYER[L * 54 + k]) {
          const a = M[0] * cx + M[1] * cy + M[2] * cz, b = M[3] * cx + M[4] * cy + M[5] * cz, c = M[6] * cx + M[7] * cy + M[8] * cz;
          cx = a; cy = b; cz = c;
          const d = M[0] * nx + M[1] * ny + M[2] * nz, e = M[3] * nx + M[4] * ny + M[5] * nz, g = M[6] * nx + M[7] * ny + M[8] * nz;
          nx = d; ny = e; nz = g;
        }
        const X = r[ro] * cx + r[ro + 1] * cy + r[ro + 2] * cz + tx;
        const Y = r[ro + 3] * cx + r[ro + 4] * cy + r[ro + 5] * cz + ty;
        const Z = r[ro + 6] * cx + r[ro + 7] * cy + r[ro + 8] * cz + tz;
        const ncx = r[ro] * nx + r[ro + 1] * ny + r[ro + 2] * nz;
        const ncy = r[ro + 3] * nx + r[ro + 4] * ny + r[ro + 5] * nz;
        const ncz = r[ro + 6] * nx + r[ro + 7] * ny + r[ro + 8] * nz;
        const facing = -(ncx * X + ncy * Y + ncz * Z) / Math.hypot(X, Y, Z);
        w[i * 54 + k] = Math.min(1, Math.max(0, (facing - 0.12) / 0.25));
        px[(i * 54 + k) * 2] = ((f * X / Z - roi[0]) / roi[2] + 0.5) * n;
        px[(i * 54 + k) * 2 + 1] = ((f * Y / Z - roi[1]) / roi[2] + 0.5) * n;
      }
    }
    return { px, w };
  }

  /** 가설 n개 x 상태 M개의 점수 (n*M). states 생략 시 지금 상태 하나. */
  score(h: Hyps, prep: Prepared, roi: Roi, states: Uint8Array[] = [this.colors]): Float64Array {
    const { n, logp, heat } = prep.maps;
    const nn = n * n;
    const { px, w } = this.project(h, roi, n);
    const M = states.length;
    const out = new Float64Array(h.n * M);
    const logP0 = Math.log(P0);
    const { xy, col, w: pw, n: P } = prep.peaks;
    const cell = new Int32Array(54), inside = new Uint8Array(54), hv = new Float64Array(54);
    for (let i = 0; i < h.n; i++) {
      for (let k = 0; k < 54; k++) {
        const x = px[(i * 54 + k) * 2], y = px[(i * 54 + k) * 2 + 1];
        const xi = Math.floor(x), yi = Math.floor(y);
        inside[k] = xi >= 0 && xi < n && yi >= 0 && yi < n ? 1 : 0;
        cell[k] = Math.min(Math.max(yi, 0), n - 1) * n + Math.min(Math.max(xi, 0), n - 1);
        const fx = x - 0.5, fy = y - 0.5;
        const x0 = Math.min(Math.max(Math.floor(fx), 0), n - 2), y0 = Math.min(Math.max(Math.floor(fy), 0), n - 2);
        const ax = Math.min(Math.max(fx - x0, 0), 1), ay = Math.min(Math.max(fy - y0, 0), 1);
        hv[k] = (heat[y0 * n + x0] * (1 - ax) + heat[y0 * n + x0 + 1] * ax) * (1 - ay) + (heat[(y0 + 1) * n + x0] * (1 - ax) + heat[(y0 + 1) * n + x0 + 1] * ax) * ay;
      }
      for (let m = 0; m < M; m++) {
        const st = states[m];
        let s = 0;
        for (let k = 0; k < 54; k++) {
          const wk = w[i * 54 + k];
          if (wk === 0) continue;
          let term: number;
          if (!inside[k]) term = OUTSIDE;
          else {
            const c = prep.cls[cell[k]];
            if (c === OTHER) term = OCCLUDED;
            else {
              const exp = st[k] === UNKNOWN ? prep.stickerLp[cell[k]] : logp[st[k] * nn + cell[k]];
              term = Math.min(Math.max(exp - logP0, WRONG), MATCH_MAX);
              if (c === BODY) term = Math.max(term, ON_BODY);
              term += HEAT_BONUS * hv[k];
            }
          }
          s += wk * term;
        }
        // 화면 → 가설: 찾은 스티커마다 같은 색으로 투영된 가장 가까운 칸
        for (let p = 0; p < P; p++) {
          let nearest = Infinity;
          const qx = xy[p * 2], qy = xy[p * 2 + 1], pc = col[p];
          for (let k = 0; k < 54; k++) {
            if (w[i * 54 + k] <= 0.05 || (st[k] !== pc && st[k] !== UNKNOWN)) continue;
            const dx = px[(i * 54 + k) * 2] - qx, dy = px[(i * 54 + k) * 2 + 1] - qy;
            const d2 = dx * dx + dy * dy;
            if (d2 < nearest) nearest = d2;
          }
          s -= UNEXPLAINED * (1 - Math.exp(-nearest / (2 * PEAK_SIGMA ** 2))) * pw[p];
        }
        out[i * M + m] = s;
      }
    }
    return out;
  }

  private current(): Hyps {
    const h = makeHyps(1);
    h.R.set(this.R!);
    h.T.set(this.Tv!);
    h.layer[0] = this.layer;
    h.phi[0] = this.phi;
    return h;
  }

  /** 보이는 칸 가운데 기대 색과 일치하는 비율 (색을 모르는 칸은 스티커이기만 하면 일치) */
  qualityOf(prep: Prepared, roi: Roi): number {
    const n = prep.maps.n;
    const { px, w } = this.project(this.current(), roi, n);
    let num = 0, den = 0;
    for (let k = 0; k < 54; k++) {
      const xi = Math.floor(px[k * 2]), yi = Math.floor(px[k * 2 + 1]);
      if (xi < 0 || xi >= n || yi < 0 || yi >= n) continue;
      const c = prep.cls[yi * n + xi];
      if (c === OTHER) continue;
      den += w[k];
      if (this.colors[k] === UNKNOWN ? c < 6 : c === this.colors[k]) num += w[k];
    }
    return den > 1.5 ? num / den : 0;
  }

  // --- 탐색 --------------------------------------------------------------------

  private perturb(R: Float64Array, Tv: Float64Array, n: number, sigDeg: number, sigT: number, sigS: number, rng = this.rand): Hyps {
    const h = makeHyps(n);
    const D = new Float64Array(9);
    for (let i = 0; i < n; i++) {
      if (i === 0) {
        h.R.set(R, 0);
        h.T.set(Tv, 0);
        continue;
      }
      rotvec(gauss(rng) * sigDeg * DEG, gauss(rng) * sigDeg * DEG, gauss(rng) * sigDeg * DEG, D);
      mul3(D, 0, R, 0, h.R, i * 9);
      const sc = Math.exp(gauss(rng) * sigS);
      h.T[i * 3] = (Tv[0] + gauss(rng) * sigT) * sc;
      h.T[i * 3 + 1] = (Tv[1] + gauss(rng) * sigT) * sc;
      h.T[i * 3 + 2] = Tv[2] * sc;
    }
    return h;
  }

  private inertia(R0: Float64Array, T0: Float64Array, h: Hyps, i: number): number {
    let tr = 0;
    for (let j = 0; j < 9; j++) tr += R0[j] * h.R[i * 9 + j];
    const angle = Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) / DEG;
    const scale = Math.log(h.T[i * 3 + 2] / T0[2]);
    const sx = h.T[i * 3] / h.T[i * 3 + 2] - T0[0] / T0[2], sy = h.T[i * 3 + 1] / h.T[i * 3 + 2] - T0[1] / T0[2];
    const shift = Math.hypot(sx, sy) * T0[2];
    return INERTIA_ROT * angle ** 2 + INERTIA_T * shift ** 2 + INERTIA_S * scale ** 2;
  }

  /** 돌아가는 층의 각도를 넓게 훑는다 (90도 닮은꼴에 갇히지 않도록). 층이 돈 경우와 몸통이 돈 경우 둘 다. */
  private sweepAngle(prep: Prepared, roi: Roi): void {
    const ds: number[] = [];
    for (let a = -66; a <= 66; a += 6) ds.push(a * DEG);
    const D = ds.length;
    const h = makeHyps(2 * D);
    const L = this.layer;
    const Rm = new Float64Array(9);
    for (let j = 0; j < 2 * D; j++) {
      const d = ds[j % D];
      h.phi[j] = this.phi + d;
      h.layer[j] = L;
      h.T.set(this.Tv!, j * 3);
      if (j < D) h.R.set(this.R!, j * 9);
      else {
        rotvec(d * AXES[L * 3], d * AXES[L * 3 + 1], d * AXES[L * 3 + 2], Rm);
        mul3(this.R!, 0, Rm, 0, h.R, j * 9);
      }
    }
    const s = this.score(h, prep, roi);
    let best = 0;
    for (let j = 0; j < 2 * D; j++) {
      if (j >= D) s[j] -= INERTIA_ROT * (ds[j - D] / DEG) ** 2 * 0.25;
      if (s[j] > s[best]) best = j;
    }
    this.R = h.R.slice(best * 9, best * 9 + 9);
    this.phi = h.phi[best];
  }

  static ROUNDS: [number, number, number, number][] = [[9.0, 0.2, 0.05, 6.0], [4.0, 0.09, 0.02, 3.0], [1.5, 0.035, 0.008, 1.5], [0.6, 0.015, 0.003, 0.6]];

  private refine(prep: Prepared, roi: Roi, rounds = Tracker.ROUNDS, n = 160, useInertia = true): number {
    const R0 = this.R!.slice(), T0 = this.Tv!.slice();
    if (this.layer >= 0) this.sweepAngle(prep, roi);
    let best = -Infinity;
    for (const [sd, st, ss, sp] of rounds) {
      const h = this.perturb(this.R!, this.Tv!, n, sd, st, ss);
      h.layer.fill(this.layer);
      h.phi.fill(this.phi);
      if (this.layer >= 0) for (let i = 1; i < n; i++) h.phi[i] += gauss(this.rand) * sp * DEG;
      const s = this.score(h, prep, roi);
      let bi = 0;
      for (let i = 0; i < n; i++) {
        if (useInertia) s[i] -= this.inertia(R0, T0, h, i);
        if (s[i] > s[bi]) bi = i;
      }
      best = s[bi];
      this.R = h.R.slice(bi * 9, bi * 9 + 9);
      this.Tv = h.T.slice(bi * 3, bi * 3 + 3);
      this.phi = h.phi[bi];
    }
    return best;
  }

  /** 여섯 층 가운데 하나가 돌기 시작했는지 떠본다 (층이 도는 경우 / 층을 잡고 몸통이 도는 경우) */
  private probeTurn(base: number, prep: Prepared, roi: Roi): void {
    const angles = [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75].map((a) => a * DEG);
    const reps = 2;
    const N = 6 * angles.length * 2 * reps;
    const h = this.perturb(this.R!, this.Tv!, N, 2.5, 0.05, 0.012);
    const held = new Uint8Array(N);
    const Rm = new Float64Array(9);
    for (let i = 0; i < N; i++) {
      const L = Math.floor(i / (angles.length * 2 * reps));
      const a = angles[Math.floor(i / (2 * reps)) % angles.length];
      h.layer[i] = L;
      h.phi[i] = a;
      held[i] = Math.floor(i / reps) % 2;
      if (i % reps === 0) {
        h.R.set(this.R!, i * 9);
        h.T.set(this.Tv!, i * 3);
      }
      if (held[i]) {
        rotvec(a * AXES[L * 3], a * AXES[L * 3 + 1], a * AXES[L * 3 + 2], Rm);
        mul3(h.R, i * 9, Rm, 0, h.R, i * 9);
      }
    }
    const s = this.score(h, prep, roi);
    let bi = 0;
    for (let i = 0; i < N; i++) {
      if (!held[i]) s[i] -= this.inertia(this.R!, this.Tv!, h, i);
      if (s[i] > s[bi]) bi = i;
    }
    if (s[bi] - base > this.turnStartGain) {
      const body = this.bodyOnly(prep, roi);
      if (this.turnVsBody > 0 && s[bi] - body.score < this.turnVsBody) {
        // 큐브가 통째로 돈 것으로도 거의 같게 설명된다: 층 회전으로 보지 않는다
        if (this.adoptBody && body.score - base > TURN_START_GAIN) this.R = body.R;
        this.rejectedTurns++;
        return;
      }
      this.turnStart = { held: held[bi] === 1, frame: this.frame, gain: s[bi] - base, bodyGain: body.score - base };
      this.R = h.R.slice(bi * 9, bi * 9 + 9);
      this.Tv = h.T.slice(bi * 3, bi * 3 + 3);
      this.layer = h.layer[bi];
      this.phi = h.phi[bi];
      this.still = this.idle = 0;
    }
  }

  /** 층 L의 21칸이 지금 자세에서 얼마나 보이나 (보이는 정도 0~1의 합) */
  layerSeen(L: number, roi: Roi): number {
    if (!this.R || !this.Tv) return 0;
    const c = this.current();
    c.layer[0] = -1;
    const { w } = this.project(c, roi, 1);
    let s = 0;
    for (let k = 0; k < 54; k++) if (LAYER[L * 54 + k]) s += w[k];
    return s;
  }

  /** 확정한 회전 검증: '돌리기 전 상태'가 뚜렷이 더 잘 맞는 프레임이 쌓이면 되돌린다 */
  private checkRevert(prep: Prepared, roi: Roi): Event | null {
    const v = this.verify!;
    const n = 24;
    const h = this.perturb(this.R!, this.Tv!, n, 2.0, 0.04, 0.01, this.rand2);
    const tab = this.score(h, prep, roi, [this.colors, v.before]);
    let cur = -Infinity, prev = -Infinity, bi = 0;
    for (let i = 0; i < n; i++) {
      if (tab[i * 2] > cur) cur = tab[i * 2];
      if (tab[i * 2 + 1] > prev) { prev = tab[i * 2 + 1]; bi = i; }
    }
    if (prev - cur > this.revertMargin) v.wins++;
    v.left--;
    if (v.wins >= this.revertWins) {
      this.verify = null;
      this.colors = v.before;
      this.log.pop();
      this.R = h.R.slice(bi * 9, bi * 9 + 9);
      this.Tv = h.T.slice(bi * 3, bi * 3 + 3);
      this.reverts++;
      const k = v.event.quarterTurns;
      const inv: Event = { center: v.event.center, quarterTurns: k === 2 ? 2 : 4 - k, frame: this.frame, info: v.event.info ? { ...v.event.info, via: "revert" } : undefined };
      this.events.push(inv);
      return inv;
    }
    if (v.left <= 0) this.verify = null;
    return null;
  }

  /** 층 없이 몸통만 돈 가설(bodyBank) 가운데 가장 잘 맞는 것 (추적 난수를 쓰지 않는다) */
  private bodyOnly(prep: Prepared, roi: Roi): { score: number; R: Float64Array } {
    const N = this.bodyBank.length / 9;
    const h = makeHyps(N);
    for (let i = 0; i < N; i++) {
      mul3(this.R!, 0, this.bodyBank, i * 9, h.R, i * 9);
      h.T.set(this.Tv!, i * 3);
    }
    const s = this.score(h, prep, roi);
    let bi = 0;
    for (let i = 1; i < N; i++) if (s[i] > s[bi]) bi = i;
    return { score: s[bi], R: h.R.slice(bi * 9, bi * 9 + 9) };
  }

  /** 가만히 있는 지금 화면을 가장 잘 설명하는 상태를 고른다 (파이썬 _decide 와 같다) */
  private decide(prep: Prepared, roi: Roi, tracked: [number, number] | null, via: "snap" | "cancel" | "doubt" = "doubt"): Event | null {
    const n = 48;
    const h = this.perturb(this.R!, this.Tv!, n, 3.0, 0.06, 0.015);
    const states: Uint8Array[] = [this.colors, ...CAND_PERMS.map((p) => permute(this.colors, p))];
    const maxBack = tracked !== null ? 0 : Math.min(HISTORY, this.log.length);
    for (let back = 1; back <= maxBack; back++) {
      for (const cp of CAND_PERMS) {
        let c = permute(this.log[this.log.length - back].before, cp);
        for (const e of this.log.slice(this.log.length - back)) c = permute(c, e.perm);
        states.push(c);
      }
    }
    const M = states.length;
    const table = this.score(h, prep, roi, states);
    const s = new Float64Array(M).fill(-Infinity);
    for (let i = 0; i < n; i++) for (let m = 0; m < M; m++) if (table[i * M + m] > s[m]) s[m] = table[i * M + m];
    const order = Array.from({ length: M - 1 }, (_, j) => j).sort((a, b) => s[1 + b] - s[1 + a]);
    const best = order[0], second = order[1];
    let choice: number | null = null;
    if (tracked !== null) {
      const t = (tracked[0] * 3 + tracked[1] - 1) * 2;
      if (best !== t && s[1 + best] - s[1 + t] > RECOVER_LEAD) choice = best;
      else if (s[1 + t] >= s[0] - TIE) choice = t;
      const relaxed = this.relaxWithin > 0 && this.frame - this.lastMoveFrame <= this.relaxWithin;
      const need = relaxed ? -Infinity : choice !== t ? Math.max(this.snapVsNone, this.snapOtherVsNone) : this.snapVsNone;
      if (choice !== null && s[1 + choice] - s[0] < need) {
        choice = null; // 고른 상태가 '회전 없음'보다 뚜렷이 낫지 않다
        this.rejectedTurns++;
      }
    } else if (s[1 + best] - s[0] > RECOVER_MARGIN && s[1 + best] - s[1 + second] > RECOVER_LEAD) choice = best;
    this.layer = -1;
    this.phi = 0;
    this.still = 0;
    if (choice === null) return null;

    const K = CAND_PERMS.length;
    const back = Math.floor(choice / K), j = choice % K;
    const [face, k] = CAND_EVENTS[j];
    const later = this.log.slice(this.log.length - back).map((e) => e.perm);
    const before = back ? this.log[this.log.length - back].before : this.colors;
    const ts = this.turnStart;
    const recent = this.bodyHist.slice(-3);
    const event: Event = {
      center: before[face * 9 + 4], quarterTurns: k, frame: this.frame,
      info: {
        via: via === "snap" && tracked !== null && choice !== (tracked[0] * 3 + tracked[1] - 1) * 2 ? "snapOther" : via,
        back, held: ts ? ts.held : null, turnFrames: ts ? this.frame - ts.frame : 0, q: this.quality,
        body: recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0,
        gain: ts ? ts.gain : 0, bodyGain: ts ? ts.bodyGain : 0, vsNone: s[1 + choice] - s[0], seen: this.layerSeen(face, roi),
      },
    };
    this.log = this.log.slice(0, this.log.length - back);
    let state = before;
    for (const perm of [CAND_PERMS[j], ...later]) {
      this.log.push({ before: state, perm });
      state = permute(state, perm);
    }
    this.log = this.log.slice(-HISTORY);
    this.colors = state;
    this.events.splice(this.events.length - back, 0, event);
    // 증거가 충분했던 확정만 '이어 돌리는 중' 창을 연다 (느슨하게 받은 회전이 다음 회전을 또 느슨하게 받게 하지 않도록)
    if (s[1 + choice] - s[0] >= this.snapVsNone) this.lastMoveFrame = this.frame;
    this.verify = back === 0 && this.revertFrames > 0 ? { left: this.revertFrames, wins: 0, before, event, frame: this.frame } : null;
    let bi = 0;
    for (let i = 0; i < n; i++) if (table[i * M + 1 + choice] > table[bi * M + 1 + choice]) bi = i;
    this.R = h.R.slice(bi * 9, bi * 9 + 9);
    this.Tv = h.T.slice(bi * 3, bi * 3 + 3);
    return event;
  }

  /** 큐브로 보이는 영역('그 밖'이 아닌 칸)의 무게중심과 어림 크기 — 화면 좌표(프레임 너비 기준) */
  private blob(prep: Prepared, roi: Roi): { u: number; v: number; side: number } | null {
    const { n } = prep.maps;
    let cnt = 0, sx = 0, sy = 0;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        // 파이썬: exp(logp[:7]).sum > 0.5 — '그 밖'일 확률 < 0.5 와 같다
        if (prep.maps.logp[OTHER * n * n + y * n + x] < Math.log(0.5)) { cnt++; sx += x; sy += y; }
      }
    if (cnt < 12) return null;
    return {
      u: ((sx / cnt + 0.5) / n - 0.5) * roi[2] + roi[0],
      v: ((sy / cnt + 0.5) / n - 0.5) * roi[2] + roi[1],
      side: (Math.sqrt(cnt / 1.35) / n) * roi[2],
    };
  }

  /** 화면 좌표(u, v)와 큐브 한 변 길이에서 평행이동 T */
  private tFrom(u: number, v: number, side: number): Float64Array {
    const tz = this.f / (side / 3);
    return Float64Array.of((u * tz) / this.f, (v * tz) / this.f, tz);
  }

  /**
   * (follow.jump) 큐브 영역 무게중심이 자세 중심에서 크게 어긋났으면(JUMP_OFF) 그쪽으로 평행이동한 자세를
   * 견줘 보고, 뚜렷이 나을 때만 출발 자세를 옮긴다. 흔들기의 관성 벌점은 옮긴 자세를 기준으로 걸린다.
   */
  private jumpStart(prep: Prepared, roi: Roi): void {
    const T0 = this.Tv!;
    const b = this.blob(prep, roi);
    if (!b) return;
    const unit = this.f / T0[2];
    const off = Math.hypot(b.u - (this.f * T0[0]) / T0[2], b.v - (this.f * T0[1]) / T0[2]) / unit;
    if (off < JUMP_OFF) return;
    const cand = Float64Array.of((b.u * T0[2]) / this.f, (b.v * T0[2]) / this.f, T0[2]);
    const h = makeHyps(2);
    [T0, cand].forEach((t, i) => {
      h.R.set(this.R!, i * 9);
      h.T.set(t, i * 3);
      h.layer[i] = this.layer;
      h.phi[i] = this.phi;
    });
    const s = this.score(h, prep, roi);
    this.diag.jumpGain = s[1] - s[0];
    if (s[1] - s[0] > JUMP_GAIN) this.Tv = cand;
  }

  /**
   * (follow.relock) 놓친 직후: 전체 탐색 전에 직전 자세 둘레만 넓게 훑는다 (전체 탐색의 6분의 1쯤).
   * 큐브를 옮겼을 뿐이면 방향은 거의 그대로이므로 이것으로 대부분 다시 잡힌다.
   */
  private localRelock(prep: Prepared, roi: Roi): boolean {
    if (!this.lastR || !this.lastT) return false;
    const b = this.blob(prep, roi);
    if (!b) return false;
    const base = this.tFrom(b.u, b.v, b.side);
    if (!(base[2] > 1 && base[2] < 400)) return false;
    const S = RELOCK_SCALES.length;
    const h = makeHyps(RELOCK_N * S);
    const D = new Float64Array(9);
    for (let i = 0; i < RELOCK_N; i++) {
      if (i === 0) D.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      else rotvec(gauss(this.rand) * RELOCK_ROT_SIG * DEG, gauss(this.rand) * RELOCK_ROT_SIG * DEG, gauss(this.rand) * RELOCK_ROT_SIG * DEG, D);
      for (let j = 0; j < S; j++) {
        const k = i * S + j;
        mul3(D, 0, this.lastR, 0, h.R, k * 9);
        const tz = base[2] / RELOCK_SCALES[j];
        h.T.set([(base[0] / base[2]) * tz, (base[1] / base[2]) * tz, tz], k * 3);
      }
    }
    this.layer = -1;
    this.phi = 0;
    const s = this.score(h, prep, roi);
    const top = Array.from(s.keys()).sort((a, b2) => s[b2] - s[a]).slice(0, 3);
    let best = -Infinity, bR: Float64Array | null = null, bT: Float64Array | null = null;
    for (const i of top) {
      this.R = h.R.slice(i * 9, i * 9 + 9);
      this.Tv = h.T.slice(i * 3, i * 3 + 3);
      const sc = this.refine(prep, roi, [[7.0, 0.25, 0.05, 0.0], ...Tracker.ROUNDS], 160, false);
      if (sc > best) { best = sc; bR = this.R; bT = this.Tv; }
    }
    this.R = bR;
    this.Tv = bT;
    if (this.qualityOf(prep, roi) < RELOCK_QUALITY) {
      this.R = this.Tv = null;
      return false;
    }
    this.localRelocks++;
    return true;
  }

  /** 처음이거나 놓쳤을 때: 큐브로 보이는 영역에서 위치·크기를 어림하고 모든 방향을 훑는다 */
  globalInit(prep: Prepared, roi: Roi): boolean {
    const b = this.blob(prep, roi);
    if (!b) return false;
    this.inits++;
    const { u, v, side } = b;
    const B = this.bank.length / 9;
    const scales = [0.8, 1.0, 1.25, 1.55];
    const all = makeHyps(B * scales.length);
    scales.forEach((sc, si) => {
      const tz = this.f / ((side * sc) / 3);
      for (let b = 0; b < B; b++) {
        const i = si * B + b;
        all.R.set(this.bank.subarray(b * 9, b * 9 + 9), i * 9);
        all.T.set([(u * tz) / this.f, (v * tz) / this.f, tz], i * 3);
      }
    });
    this.layer = -1;
    this.phi = 0;
    const s = this.score(all, prep, roi);
    const top = Array.from(s.keys()).sort((a, b) => s[b] - s[a]).slice(0, 6);
    let best = -Infinity, bR = this.R, bT = this.Tv;
    for (const i of top) {
      this.R = all.R.slice(i * 9, i * 9 + 9);
      this.Tv = all.T.slice(i * 3, i * 3 + 3);
      const sc = this.refine(prep, roi, [[14.0, 0.5, 0.1, 0.0], [7.0, 0.25, 0.05, 0.0], ...Tracker.ROUNDS], 160, false);
      if (sc > best) { best = sc; bR = this.R; bT = this.Tv; }
    }
    this.R = bR;
    this.Tv = bT;
    return true;
  }

  // --- 한 프레임 ----------------------------------------------------------------

  update(maps: Maps, roi: Roi): Event[] {
    this.frame++;
    const prep = prepare(maps);
    this.prep = prep;
    const out: Event[] = [];
    if (!this.R) {
      // 옮기기만 했을 가능성이 크므로 직전 자세 둘레부터 싸게 훑고(follow.relock), 안 되면 전체 탐색
      if (this.follow.relock && this.localRelock(prep, roi)) {
        this.quality = this.qualityOf(prep, roi);
        this.settle = this.settleFrames;
        this.lostFrames = 0;
        this.seek = null;
        if (this.follow.calmAfterLock) this.calm = CALM_AFTER + 1;
        return out;
      }
      const b = this.follow.wide ? this.blob(prep, roi) : null;
      if (this.globalInit(prep, roi)) {
        this.quality = this.qualityOf(prep, roi);
        if (this.quality < 0.5) this.R = this.Tv = null; // 자신 없는 초기화는 버리고 다음 프레임에 다시
        else this.settle = this.settleFrames;
      }
      if (this.R) {
        this.lostFrames = 0;
        this.seek = null;
        if (this.follow.calmAfterLock && this.lastR) this.calm = CALM_AFTER + 1; // 놓쳤다 다시 잡은 경우만
      } else if (this.follow.wide) {
        // 큐브로 보이는 것을 봤으면 다음에는 그쪽으로 좁혀 찾는다 (같은 곳은 몇 번까지만)
        const tries = this.seek ? this.seek.tries + 1 : 1;
        this.seek = b && tries <= SEEK_TRIES ? { u: b.u, v: b.v, side: b.side, tries } : null;
        this.lostFrames++;
      }
      return out;
    }
    this.diag.jumpGain = 0;
    // 옮기는 중이면 한 프레임 앞을 내다본 자리에서 출발해 넓게 흔든다 (관성 벌점도 그 자리 기준)
    this.moving = this.isMoving();
    if (this.moving) this.Tv = Float64Array.of(this.Tv![0] + this.vel[0], this.Tv![1] + this.vel[1], this.Tv![2]);
    // 큐브를 빠르게 옮기는 동안(과 멈춘 직후)은 자세가 흔들려 가짜 회전이 나오기 쉽다: 새 회전 찾기·의심 판정을 쉰다
    this.calm = this.moving ? CALM_AFTER : Math.max(0, this.calm - 1);
    const hold = this.moving || this.calm > 0;
    if (this.follow.jump) this.jumpStart(prep, roi);
    const prevR = this.R.slice();
    let base = this.refine(prep, roi, this.moving ? [WIDE_ROUND, ...Tracker.ROUNDS] : Tracker.ROUNDS);
    if (this.settleFrames > 0 && this.layer < 0) {
      let tr = 0;
      for (let j = 0; j < 9; j++) tr += prevR[j] * this.R![j];
      const bodyDeg = Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) / DEG;
      if (bodyDeg > FAST_BODY_DEG) this.settle = this.settleFrames; // 통째로 빠르게 도는 중
      else if (this.settle > 0 && this.quality >= SETTLE_QUALITY) this.settle--;
    }
    const settling = this.settle > 0;
    if (this.layer >= 0) {
      const k = Math.round(this.phi / (Math.PI / 2));
      const resid = this.phi - (k * Math.PI) / 2;
      const kk = ((k % 4) + 4) % 4;
      if (kk && Math.abs(resid) < SNAP) {
        // 90도 단위에 닿았다: 멈추기를 기다리지 않고 바로 확정 (빠르게 이어 돌리면 멈춘 프레임이 없다)
        const ev = this.decide(prep, roi, [this.layer, kk], "snap");
        this.turnStart = null;
        if (ev) out.push(ev);
        const h = this.current();
        base = this.score(h, prep, roi)[0];
      } else if (k === 0 && Math.abs(this.phi) < CANCEL) {
        this.idle++;
        if (this.idle >= 3) {
          this.layer = -1;
          this.phi = 0;
          this.idle = 0;
          const ev = this.decide(prep, roi, null, "cancel");
          if (ev) out.push(ev);
          this.turnStart = null;
        }
      } else this.idle = 0;
    }
    if (this.verify && this.layer < 0 && this.frame > this.verify.frame) {
      const ev = this.checkRevert(prep, roi);
      if (ev) out.push(ev);
    }
    if (this.layer < 0 && this.allowTurns && !settling && !hold) this.probeTurn(base, prep, roi);

    this.quality = this.qualityOf(prep, roi);
    this.doubt = this.layer < 0 && this.quality < DOUBT_QUALITY ? this.doubt + 1 : 0;
    if (this.doubt >= DOUBT_FRAMES && !this.colors.includes(UNKNOWN) && !settling && !hold) {
      const ev = this.decide(prep, roi, null, "doubt");
      if (ev) {
        out.push(ev);
        this.doubt = 0;
        this.quality = this.qualityOf(prep, roi);
      }
    }
    this.badFrames = this.quality < LOST_QUALITY ? this.badFrames + 1 : 0;
    this.zeroRun = this.quality === 0 ? this.zeroRun + 1 : 0;
    this.coverage = this.coverageOf(prep, roi);
    this.lowCover = this.useCover && this.layer < 0 && this.coverage < COVER_MIN ? this.lowCover + 1 : 0;
    this.measure(prep, roi);
    // 몸통 회전 속도 (진단용)
    if (this.prevRend && this.R) {
      let tr = 0;
      for (let j = 0; j < 9; j++) tr += this.prevRend[j] * this.R[j];
      this.diag.bodyDeg = Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) / DEG;
      this.bodyHist.push(this.diag.bodyDeg);
      if (this.bodyHist.length > 5) this.bodyHist.shift();
    }
    this.prevRend = this.R ? this.R.slice() : null;
    // 일치율이 정확히 0이면 자세가 큐브 위에 아예 없다 (가려져서 낮은 것과 다르다): 기다리지 않고 다시 찾는다
    const zeroLost = this.follow.fastLost && this.zeroRun >= ZERO_LOST;
    if (this.badFrames >= LOST_FRAMES || zeroLost || this.lowCover >= COVER_FRAMES) {
      if (this.lowCover >= COVER_FRAMES) this.relocks++;
      this.reset(); // 놓쳤다: 다음 프레임에 넓게 다시 찾는다 (상태는 그대로)
    } else if (this.follow.predict && this.Tv) {
      // 자세 중심 속도 (튐에 끌려가지 않게 눌러서 잰다)
      if (this.prevT) {
        for (let j = 0; j < 3; j++) this.vel[j] = VEL_SMOOTH * (this.Tv[j] - this.prevT[j]) + (1 - VEL_SMOOTH) * this.vel[j];
        const m = Math.hypot(this.vel[0], this.vel[1]);
        if (m > VEL_MAX) for (let j = 0; j < 2; j++) this.vel[j] *= VEL_MAX / m;
      }
      this.prevT = this.Tv.slice();
    }
    return out;
  }

  /** 진단값: 큐브 영역 무게중심이 자세 중심에서 얼마나 떨어졌나, 자세 중심이 한 프레임에 얼마나 움직였나 (조각 단위) */
  private measure(prep: Prepared, roi: Roi): void {
    const T0 = this.Tv!;
    const unit = this.f / T0[2];
    const u = (this.f * T0[0]) / T0[2], v = (this.f * T0[1]) / T0[2];
    const b = this.blob(prep, roi);
    this.diag.blobOff = b ? Math.hypot(b.u - u, b.v - v) / unit : -1;
    const P = this.diagPrev;
    this.diag.speed = P ? Math.hypot(u - P[0], v - P[1]) / unit : 0;
    this.diagPrev = [u, v];
  }
  private diagPrev: [number, number] | null = null;

  /** 스티커로 읽힌 칸 가운데, 지금 자세에서 보이는 스티커 자리(칸 간격의 0.75배 안)로 설명되는 비율 */
  coverageOf(prep: Prepared, roi: Roi): number {
    if (!this.R || !this.Tv) return 0;
    const n = prep.maps.n;
    const { px, w } = this.project(this.current(), roi, n);
    const spacing = ((this.f / this.Tv[2]) / roi[2]) * n; // 조각 한 변의 지도 칸 수
    const r2 = (0.75 * spacing) ** 2;
    const pts: number[] = [];
    for (let k = 0; k < 54; k++) if (w[k] > 0.05) pts.push(px[k * 2], px[k * 2 + 1]);
    let tot = 0, hit = 0;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        if (prep.cls[y * n + x] >= 6) continue;
        tot++;
        const cx = x + 0.5, cy = y + 0.5;
        for (let j = 0; j < pts.length; j += 2) {
          const dx = pts[j] - cx, dy = pts[j + 1] - cy;
          if (dx * dx + dy * dy < r2) { hit++; break; }
        }
      }
    return tot >= 20 ? hit / tot : 1;
  }

  /** 마지막 프레임 기준으로, 지금 자세에서 보이는 칸마다 읽은 색 (스캔용) */
  readings(roi: Roi): { k: number; w: number; cls: number; p: number }[] {
    if (!this.R || !this.prep) return [];
    const prep = this.prep;
    const n = prep.maps.n, nn = n * n;
    const { px, w } = this.project(this.current(), roi, n);
    const out: { k: number; w: number; cls: number; p: number }[] = [];
    for (let k = 0; k < 54; k++) {
      const xi = Math.floor(px[k * 2]), yi = Math.floor(px[k * 2 + 1]);
      if (xi < 1 || xi >= n - 1 || yi < 1 || yi >= n - 1 || w[k] < 0.5) continue;
      const c = prep.cls[yi * n + xi];
      out.push({ k, w: w[k], cls: c, p: Math.exp(prep.maps.logp[c * nn + yi * n + xi]) });
    }
    return out;
  }

  /** 화면에서 본 각 칸의 위치 (영역 안 0~1 좌표) — 화면에 겹쳐 그리기용 */
  overlay(roi: Roi): { k: number; x: number; y: number; w: number }[] {
    if (!this.R) return [];
    const { px, w } = this.project(this.current(), roi, 1);
    const out = [];
    for (let k = 0; k < 54; k++) if (w[k] > 0.05) out.push({ k, x: px[k * 2], y: px[k * 2 + 1], w: w[k] });
    return out;
  }

  /**
   * 사람이 보는 방향(코치의 "보는 방향"): 위를 향한 면과 사람 쪽을 향한 면의 가운데 색.
   * 카메라는 사람 맞은편에 있으므로 사람 쪽 = 카메라에서 멀어지는 쪽(카메라 z +)이다.
   */
  view(): { up: number; front: number } | null {
    if (!this.R) return null;
    const R = this.R;
    let up = -1, front = -1, bu = -Infinity, bf = -Infinity;
    for (let slot = 0; slot < 6; slot++) {
      const k = slot * 9 + 4;
      const nx = NORMALS[k * 3], ny = NORMALS[k * 3 + 1], nz = NORMALS[k * 3 + 2];
      const cy = R[3] * nx + R[4] * ny + R[5] * nz; // 카메라 y는 아래쪽
      const cz = R[6] * nx + R[7] * ny + R[8] * nz; // 카메라 z는 보는 방향
      if (-cy > bu) { bu = -cy; up = slot; }
      if (cz > bf) { bf = cz; front = slot; }
    }
    if (up === front) return null;
    return { up: this.colors[up * 9 + 4], front: this.colors[front * 9 + 4] };
  }

  /** 지금 돌아가는 층의 가운데 색과 각도 (라디안, 바깥에서 봐서 시계 방향 +) */
  turning(): { color: number; angle: number } | null {
    return this.layer >= 0 ? { color: this.colors[this.layer * 9 + 4], angle: this.phi } : null;
  }
}

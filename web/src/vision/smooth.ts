/**
 * 화면에 그리는 자세만 부드럽게 한다 (추적기 내부 값은 그대로 둔다).
 *
 * One-Euro 필터: 가만히 있을 때는 세게 눌러 떨림을 없애고, 빨리 움직일 때는 거의 그대로 따라가
 * 늦어 보이지 않게 한다. 차단 주파수 = minCutoff + beta x (움직이는 속도).
 */

/** 여러 값을 한 묶음으로 거르는 One-Euro 필터 (속도는 묶음 전체의 크기로 잰다) */
export class OneEuro {
  private x: Float64Array | null = null;
  private dx: Float64Array;
  private t = 0;
  private n: number;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  constructor(n: number, minCutoff: number, beta: number, dCutoff = 1.0) {
    this.n = n;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.dx = new Float64Array(n);
  }

  reset(): void {
    this.x = null;
  }

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  /** t는 초 단위. 처음이거나 reset 뒤면 그대로 낸다. */
  filter(v: ArrayLike<number>, t: number): Float64Array {
    if (!this.x) {
      this.x = Float64Array.from(v);
      this.dx.fill(0);
      this.t = t;
      return this.x.slice();
    }
    const dt = Math.min(Math.max(t - this.t, 1 / 240), 0.5);
    this.t = t;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    let speed = 0;
    for (let i = 0; i < this.n; i++) {
      const d = (v[i] - this.x[i]) / dt;
      this.dx[i] += ad * (d - this.dx[i]);
      speed += this.dx[i] ** 2;
    }
    const a = OneEuro.alpha(this.minCutoff + this.beta * Math.sqrt(speed), dt);
    for (let i = 0; i < this.n; i++) this.x[i] += a * (v[i] - this.x[i]);
    return this.x.slice();
  }
}

/** 회전 행렬(행 우선 9개) → 사원수 [w, x, y, z] */
export function matToQuat(R: ArrayLike<number>): Float64Array {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = Array.from(R);
  const tr = m00 + m11 + m22;
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = s / 4; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = s / 4; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = s / 4; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = s / 4;
  }
  return Float64Array.of(w, x, y, z);
}

/** 사원수 [w, x, y, z] → 회전 행렬 (행 우선) */
export function quatToMat(q: ArrayLike<number>): Float64Array {
  let [w, x, y, z] = Array.from(q);
  const n = Math.hypot(w, x, y, z) || 1;
  w /= n; x /= n; y /= n; z /= n;
  return Float64Array.of(
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  );
}

/** 두 회전 사이 각도 (도) */
export function angleBetween(A: ArrayLike<number>, B: ArrayLike<number>): number {
  let tr = 0;
  for (let j = 0; j < 9; j++) tr += A[j] * B[j];
  return (Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) * 180) / Math.PI;
}

/**
 * 표시용 자세 평활. 평행이동은 조각 단위(큐브 한 변 = 3), 회전은 사원수로 거른다.
 * 놓쳤다 다시 잡았을 때, 층 회전이 시작·확정될 때, 한 프레임에 크게 뛸 때는 거르지 않고 바로 옮긴다
 * (옛 자세에서 미끄러져 오면 '느리게 따라온다'로 보인다).
 */
export class PoseSmoother {
  R: Float64Array | null = null;
  T: Float64Array | null = null;
  private q: Float64Array | null = null;
  private fT: OneEuro;
  private fR: OneEuro;
  private lastRawR: Float64Array | null = null;
  private lastRawT: Float64Array | null = null;
  /** 바로 옮긴 횟수 (진단용) */
  snaps = 0;

  /**
   * 기본값은 샘플 영상의 추적 자세 흐름으로 골랐다: 꼭짓점 떨림(2차 차분) 13.8 → 6.7px,
   * 표시가 추적 자세보다 뒤처지는 정도 p90 9.7px (화면 너비 960px 기준). 더 세게 누르면 떨림은 줄지만 늦어 보인다.
   */
  constructor(opts: { tMin?: number; tBeta?: number; rMin?: number; rBeta?: number } = {}) {
    this.fT = new OneEuro(3, opts.tMin ?? 1.5, opts.tBeta ?? 1.5);
    this.fR = new OneEuro(4, opts.rMin ?? 1.5, opts.rBeta ?? 6.0);
  }

  reset(): void {
    this.R = this.T = this.q = this.lastRawR = this.lastRawT = null;
    this.fT.reset();
    this.fR.reset();
  }

  /** 새 추적 자세를 넣는다. snap=true면 거르지 않고 그대로 옮긴다. */
  update(R: ArrayLike<number>, T: ArrayLike<number>, tSec: number, snap = false): void {
    const big =
      this.lastRawR !== null && this.lastRawT !== null &&
      (angleBetween(this.lastRawR, R) > 25 || Math.hypot(T[0] - this.lastRawT[0], T[1] - this.lastRawT[1], T[2] - this.lastRawT[2]) > 2.5);
    if (snap || big || !this.q) {
      this.fT.reset();
      this.fR.reset();
      if (this.q) this.snaps++;
    }
    this.lastRawR = Float64Array.from(R);
    this.lastRawT = Float64Array.from(T);
    const q = matToQuat(R);
    // 사원수 q와 -q는 같은 회전: 직전 값과 같은 쪽 부호로 맞춰야 평균이 깨지지 않는다
    if (this.q && q[0] * this.q[0] + q[1] * this.q[1] + q[2] * this.q[2] + q[3] * this.q[3] < 0) for (let i = 0; i < 4; i++) q[i] = -q[i];
    this.q = this.fR.filter(q, tSec);
    this.R = quatToMat(this.q);
    this.T = this.fT.filter(T, tSec);
  }
}

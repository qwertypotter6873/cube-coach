/**
 * 모델 지도 크기 바꾸기: 크게 넣은 입력(256)의 64x64 지도를 추적기·면 읽기가 쓰는 48x48 로 줄인다.
 * 입력 256 실험 갈래(커밋 5154035, README 5.2절)에서 썼고, 시험셋에서 나빠 지금 앱(seg.ts, 입력 192)은 쓰지 않는다.
 * 파이썬 cubecoach.segnet.area_weights / maps_to 와 같은 계산 (web/test/maps.test.ts 가 대조한다).
 */

const NC = 8;

/** n_in 칸을 n_out 칸으로 줄일 때 칸마다 겹치는 길이 (파이썬 cubecoach.segnet.area_weights 와 같음) */
function areaWeights(nIn: number, nOut: number): { k0: number; w: number[] }[] {
  const s = nIn / nOut;
  const rows: { k0: number; w: number[] }[] = [];
  for (let i = 0; i < nOut; i++) {
    const a = i * s, b = (i + 1) * s;
    const k0 = Math.floor(a), k1 = Math.min(nIn, Math.ceil(b));
    const w: number[] = [];
    for (let k = k0; k < k1; k++) w.push(Math.max(0, Math.min(b, k + 1) - Math.max(a, k)));
    rows.push({ k0, w });
  }
  return rows;
}

/**
 * 로그확률(NC x n x n)과 중심 지도(n x n)를 nOut x nOut 으로 넓이 평균해 줄인다: 확률로 평균하고 다시 로그 (maps_to 와 같음).
 */
export function mapsTo(logp: Float32Array, heat: Float32Array, n: number, nOut: number): { logp: Float32Array; heat: Float32Array } {
  if (n === nOut) return { logp, heat };
  const W = areaWeights(n, nOut);
  const norm = (n / nOut) ** 2;
  const nn = n * n, mm = nOut * nOut;
  const lp = new Float32Array(NC * mm);
  const ht = new Float32Array(mm);
  const p = new Float64Array(NC);
  for (let i = 0; i < nOut; i++) {
    const wy = W[i];
    for (let j = 0; j < nOut; j++) {
      const wx = W[j];
      p.fill(0);
      let h = 0;
      for (let a = 0; a < wy.w.length; a++) {
        const y = wy.k0 + a;
        for (let b = 0; b < wx.w.length; b++) {
          const x = wx.k0 + b;
          const w = wy.w[a] * wx.w[b];
          const q = y * n + x;
          for (let c = 0; c < NC; c++) p[c] += w * Math.exp(logp[c * nn + q]);
          h += w * heat[q];
        }
      }
      let sum = 0;
      for (let c = 0; c < NC; c++) sum += p[c];
      const o = i * nOut + j;
      for (let c = 0; c < NC; c++) lp[c * mm + o] = Math.log(Math.max(p[c] / sum, 1e-12));
      ht[o] = h / norm;
    }
  }
  return { logp: lp, heat: ht };
}

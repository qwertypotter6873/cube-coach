/**
 * 직접 학습한 인식 모델(CubeSegNet, ONNX 664KB)을 브라우저에서 돌린다.
 *
 * 화면(카메라·영상)에서 추적기가 정한 정사각 관심영역을 잘라 192x192로 줄이고,
 * 모델이 낸 9채널 48x48 지도를 추적기 입력 모양으로 바꾼다.
 *   채널 0~7 → 칸마다 로그확률 (log-softmax)
 *   채널 8   → 스티커 중심 지도 (sigmoid)
 */

import * as ort from "onnxruntime-web/wasm";
import type { Maps, Roi } from "./tracker.ts";

const RES = 192;
const NC = 8;

export class SegModel {
  private session: ort.InferenceSession;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private input = new Float32Array(3 * RES * RES);
  lastMs = 0;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = RES;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  }

  static async load(modelUrl: string, wasmBase: string): Promise<SegModel> {
    ort.env.wasm.wasmPaths = wasmBase;
    ort.env.wasm.numThreads = 1; // 정적 호스팅(교차 출처 격리 없음)에서도 돌도록 단일 스레드
    const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return new SegModel(session);
  }

  /** 지금 잘라 넣은 영역 (화면에 겹쳐 보여 주기용) */
  get crop(): HTMLCanvasElement {
    return this.canvas;
  }

  /**
   * 화면 W x H 에서 roi(프레임 너비 기준, 중심이 0) 영역을 잘라 모델에 넣는다.
   * flip: 카메라가 좌우반전된 화면을 줄 때 true. roi는 '실제 세계' 좌표로 받고, 반전된 원본에서
   * 해당 영역을 찾아 좌우를 되돌려 넣는다 (모델·추적기는 늘 반전 안 된 화면을 본다).
   */
  async run(src: CanvasImageSource, W: number, H: number, roi: Roi, flip = false): Promise<Maps> {
    const t0 = performance.now();
    const side = roi[2] * W;
    let sx = W / 2 + roi[0] * W - side / 2;
    const sy = H / 2 + roi[1] * W - side / 2;
    if (flip) sx = W - sx - side; // 반전된 원본에서의 같은 영역
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, RES, RES);
    // 화면 밖으로 나간 부분은 검게 둔다 (브라우저마다 다른 잘림 처리에 기대지 않도록 직접 자른다)
    const x0 = Math.max(0, sx), y0 = Math.max(0, sy);
    const x1 = Math.min(W, sx + side), y1 = Math.min(H, sy + side);
    if (x1 > x0 && y1 > y0) {
      const k = RES / side;
      if (flip) ctx.setTransform(-1, 0, 0, 1, RES, 0); // 잘라 넣은 그림을 좌우로 되돌린다
      ctx.drawImage(src, x0, y0, x1 - x0, y1 - y0, (x0 - sx) * k, (y0 - sy) * k, (x1 - x0) * k, (y1 - y0) * k);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    const px = ctx.getImageData(0, 0, RES, RES).data;
    const n2 = RES * RES;
    for (let i = 0; i < n2; i++) {
      this.input[i] = px[i * 4] / 255;
      this.input[n2 + i] = px[i * 4 + 1] / 255;
      this.input[2 * n2 + i] = px[i * 4 + 2] / 255;
    }
    const out = await this.session.run({ image: new ort.Tensor("float32", this.input, [1, 3, RES, RES]) });
    const maps = out.maps.data as Float32Array;
    const n = out.maps.dims[3];
    const nn = n * n;
    const logp = new Float32Array(NC * nn);
    const heat = new Float32Array(nn);
    for (let p = 0; p < nn; p++) {
      let mx = -Infinity;
      for (let c = 0; c < NC; c++) mx = Math.max(mx, maps[c * nn + p]);
      let sum = 0;
      for (let c = 0; c < NC; c++) sum += Math.exp(maps[c * nn + p] - mx);
      const lse = mx + Math.log(sum);
      for (let c = 0; c < NC; c++) logp[c * nn + p] = maps[c * nn + p] - lse;
      heat[p] = 1 / (1 + Math.exp(-maps[NC * nn + p]));
    }
    this.lastMs = performance.now() - t0;
    return { n, logp, heat };
  }
}

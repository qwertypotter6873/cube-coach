/**
 * 카메라·영상 시작과 멈춤 (web/src/vision/camera.ts).
 *
 * 브라우저 없이 돌리려고 영상 요소·카메라 권한·인식 모델을 흉내 낸다.
 *  - 모델 불러오기와 카메라 권한은 시험이 원할 때 끝나게 한다 (기다리는 사이 멈추거나 다른 입력으로 바꾸기).
 *  - 영상 요소는 requestVideoFrameCallback 으로 예약된 콜백을 세고, 시험이 한 프레임씩 내보낸다.
 * 확인하는 것
 *  1. 기다리는 사이 멈추거나 다른 입력으로 바꾸면, 늦게 끝난 옛 시작이 되살아나지 않는다.
 *  2. 멈췄다 다시 시작해도 프레임 고리는 하나뿐이다 (예약된 콜백 1개).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { SegModel } from "../src/vision/seg.ts";
import { VisionInput } from "../src/vision/camera.ts";
import type { VisionEvents } from "../src/vision/camera.ts";

interface Deferred<T> {
  promise: Promise<T>;
  resolve(v: T): void;
  reject(e: Error): void;
}
function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void, reject!: (e: Error) => void;
  const promise = new Promise<T>((a, b) => ((resolve = a), (reject = b)));
  return { promise, resolve, reject };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

/** 영상 요소 흉내: play/pause 와 프레임 콜백 예약만 */
class FakeVideo {
  muted = false;
  playsInline = false;
  loop = false;
  ended = false;
  paused = true;
  videoWidth = 640;
  videoHeight = 360;
  currentTime = 0;
  src = "";
  srcObject: unknown = null;
  plays = 0;
  pending = new Map<number, (t: number, m: { presentedFrames: number }) => void>();
  private nextId = 1;
  private playing: Deferred<void> | null = null;
  readonly vfc: boolean;
  constructor(vfc = true) {
    this.vfc = vfc;
  }
  addEventListener(): void {}
  play(): Promise<void> {
    this.plays++;
    this.paused = false;
    // 브라우저처럼: 재생 시작 전에 pause()가 불리면 play()는 AbortError로 끝난다
    this.playing = deferred<void>();
    const d = this.playing;
    queueMicrotask(() => d.resolve());
    return d.promise;
  }
  pause(): void {
    this.paused = true;
    if (this.playing) {
      const e = new Error("The play() request was interrupted by a call to pause().");
      e.name = "AbortError";
      this.playing.reject(e);
      this.playing = null;
    }
  }
  get requestVideoFrameCallback() {
    return this.vfc ? (cb: (t: number, m: { presentedFrames: number }) => void) => this.add(cb) : undefined;
  }
  get cancelVideoFrameCallback() {
    return this.vfc ? (id: number) => void this.pending.delete(id) : undefined;
  }
  add(cb: (t: number, m: { presentedFrames: number }) => void): number {
    const id = this.nextId++;
    this.pending.set(id, cb);
    return id;
  }
  /** 새 프레임 하나: 예약된 콜백을 모두 부른다 */
  frame(): void {
    const cbs = [...this.pending.values()];
    this.pending.clear();
    for (const cb of cbs) cb(performance.now(), { presentedFrames: 0 });
  }
}

/** 그리기 문맥: 무엇을 불러도 아무 일도 하지 않는다 */
const nullCtx = new Proxy({}, { get: () => () => undefined, set: () => true });

interface Env {
  video: FakeVideo;
  vision: VisionInput;
  load: Deferred<SegModel>;
  runs: Deferred<unknown>[];
  gum: Deferred<{ getTracks(): { stop(): void }[] }>[];
  progress: number;
  /** requestAnimationFrame 흉내 (영상 프레임 콜백이 없을 때) */
  raf: Map<number, () => void>;
}

function setup(opts: { vfc?: boolean } = {}): Env {
  const video = new FakeVideo(opts.vfc ?? true);
  const env = { video, runs: [], gum: [], progress: 0, raf: new Map() } as unknown as Env;
  const g = globalThis as Record<string, unknown>;
  g.document = {
    baseURI: "http://localhost/",
    createElement: (tag: string) => (tag === "video" ? video : { getContext: () => nullCtx, width: 0, height: 0 }),
  };
  let rafId = 1;
  g.requestAnimationFrame = (cb: () => void) => (env.raf.set(rafId, cb), rafId++);
  g.cancelAnimationFrame = (id: number) => void env.raf.delete(id);
  Object.defineProperty(globalThis.navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: () => {
        const d = deferred<{ getTracks(): { stop(): void }[] }>();
        env.gum.push(d);
        return d.promise;
      },
    },
  });
  env.load = deferred<SegModel>();
  const model = {
    run: () => {
      const d = deferred<unknown>();
      env.runs.push(d);
      return d.promise;
    },
  };
  (SegModel as unknown as { load: () => Promise<unknown> }).load = () => env.load.promise;
  env.load.promise.catch(() => {});
  (env as unknown as { model: unknown }).model = model;
  const ev: VisionEvents = {
    status: () => {},
    scanProgress: () => void env.progress++,
    scanned: () => {},
    turning: () => {},
    turnCancel: () => {},
    move: () => {},
    pose: () => {},
  };
  env.vision = new VisionInput({ getContext: () => nullCtx, width: 0, height: 0 } as unknown as HTMLCanvasElement, ev);
  return env;
}
const modelOf = (env: Env) => (env as unknown as { model: SegModel }).model;
/** 예약된 프레임 콜백 수 (영상 프레임 콜백 + requestAnimationFrame) */
const scheduled = (env: Env) => env.video.pending.size + env.raf.size;
/** 비어 있는 인식 결과 (큐브가 안 보이는 화면) */
const emptyMaps = () => ({ n: 48, logp: new Float32Array(8 * 48 * 48), heat: new Float32Array(48 * 48) });

test("모델을 불러오는 중에 멈추면, 다 불러온 뒤에도 시작하지 않는다", async () => {
  const env = setup();
  const p = env.vision.startUrl("./samples/a.mp4");
  await tick();
  assert.equal(env.vision.phase, "loading");
  env.vision.stop(); // 가상 큐브로 바꾸기 (main.ts setMode)
  env.load.resolve(modelOf(env));
  await p;
  await tick();
  assert.equal(env.vision.phase, "idle", "멈춘 뒤 옛 시작이 되살아났다");
  assert.equal(env.vision.trackState, "idle");
  assert.equal(env.video.plays, 0, "영상을 재생하면 안 된다");
  assert.equal(scheduled(env), 0, "프레임 고리가 돌면 안 된다");
});

test("카메라: 권한을 허락하면 스캔을 시작하고, 멈추면 카메라를 끄고 프레임 고리도 멈춘다", async () => {
  const env = setup();
  const cam = env.vision.startCamera();
  await tick();
  assert.equal(env.vision.phase, "loading");
  env.load.resolve(modelOf(env));
  await tick();
  assert.equal(env.gum.length, 1, "모델을 불러온 뒤 권한을 묻는다");
  let stopped = 0;
  const stream = { getTracks: () => [{ stop: () => void stopped++ }] };
  env.gum[0].resolve(stream);
  await cam;
  assert.equal(env.vision.phase, "scan");
  assert.equal(env.video.srcObject, stream);
  assert.equal((env.vision as unknown as { selfieView: boolean }).selfieView, true, "카메라는 거울처럼 보여 준다");
  assert.equal(scheduled(env), 1);
  env.vision.stop();
  assert.equal(stopped, 1, "멈추면 카메라를 끈다");
  assert.equal(env.vision.phase, "idle");
  assert.equal(scheduled(env), 0);
});

test("카메라 권한을 묻는 사이 샘플 영상으로 바꾸면, 늦게 허락된 카메라는 바로 끈다", async () => {
  const env = setup();
  env.load.resolve(modelOf(env));
  const cam = env.vision.startCamera();
  await tick();
  assert.equal(env.gum.length, 1, "권한을 물어야 한다");
  const url = env.vision.startUrl("./samples/a.mp4");
  await url;
  assert.equal(env.vision.phase, "scan");
  // 이제야 권한 허락
  let stopped = 0;
  const stream = { getTracks: () => [{ stop: () => void stopped++ }] };
  env.gum[0].resolve(stream);
  await cam; // 오류 없이 조용히 끝나야 한다 (main.ts 가 오류 안내를 띄우지 않게)
  await tick();
  assert.equal(stopped, 1, "버린 카메라 스트림을 꺼야 한다");
  assert.notEqual(env.video.srcObject, stream, "버린 카메라를 영상에 붙이면 안 된다");
  assert.equal(env.video.src, "./samples/a.mp4");
  assert.equal(env.vision.phase, "scan");
  assert.equal(scheduled(env), 1);
});

test("재생을 기다리는 사이 다른 영상으로 바꾸면, 끊긴 옛 재생은 오류 없이 버린다", async () => {
  const env = setup();
  env.load.resolve(modelOf(env));
  const a = env.vision.startUrl("./samples/a.mp4");
  const b = env.vision.startUrl("./samples/b.mp4"); // a 의 play() 가 pause() 로 끊긴다
  await assert.doesNotReject(a);
  await b;
  await tick();
  assert.equal(env.video.src, "./samples/b.mp4");
  assert.equal(env.vision.phase, "scan");
  assert.equal(scheduled(env), 1, "프레임 고리는 하나뿐이어야 한다");
});

for (const vfc of [true, false]) {
  test(`멈췄다 다시 시작해도 프레임 고리는 하나다 (${vfc ? "requestVideoFrameCallback" : "requestAnimationFrame"})`, async () => {
    const env = setup({ vfc });
    env.load.resolve(modelOf(env));
    const fire = () => {
      env.video.frame();
      const cbs = [...env.raf.values()];
      env.raf.clear();
      cbs.forEach((cb) => cb());
    };
    await env.vision.startUrl("./samples/a.mp4");
    assert.equal(scheduled(env), 1);
    for (let i = 0; i < 3; i++) fire();
    assert.equal(scheduled(env), 1);
    env.vision.stop();
    assert.equal(scheduled(env), 0, "멈추면 예약된 콜백을 취소해야 한다");
    await env.vision.startUrl("./samples/a.mp4");
    assert.equal(scheduled(env), 1);
    for (let i = 0; i < 5; i++) {
      fire();
      assert.equal(scheduled(env), 1, `다시 시작한 뒤 ${i + 1}번째 프레임에서 고리가 늘었다`);
    }
  });
}

test("멈추기 전에 넣은 프레임의 인식 결과는 다시 시작한 뒤에 쓰지 않는다", async () => {
  const env = setup();
  env.load.resolve(modelOf(env));
  await env.vision.startUrl("./samples/a.mp4");
  assert.equal(env.runs.length, 1, "첫 프레임을 모델에 넣었다");
  env.vision.stop();
  await env.vision.startUrl("./samples/b.mp4");
  const before = env.progress;
  env.runs[0].resolve(emptyMaps()); // 옛 프레임 결과가 늦게 온다
  await tick();
  assert.equal(env.progress, before, "옛 프레임으로 면 읽기를 진행하면 안 된다");
  // 새 프레임은 정상으로 처리한다
  env.video.frame();
  assert.equal(env.runs.length, 2);
  env.runs[1].resolve(emptyMaps());
  await tick();
  assert.equal(env.progress, before + 1);
});

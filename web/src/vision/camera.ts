/**
 * 카메라(또는 영상 파일) → 인식 모델 → 추적기 → 코치 이벤트.
 *
 * 단계
 *  1. 스캔: 한 면씩 정면으로 보여 주면 3x3 격자를 읽어 모으고, 여섯 면이 모이면 면 방향 4^6 조합 가운데
 *     실제로 존재할 수 있는 상태를 고른다 (facescan.ts). 면을 어느 방향으로 보여 줘도 된다.
 *  2. 추적: 그 상태로 매 프레임 자세와 돌아가는 층을 따라가며, 가상 큐브(키보드)와 똑같은
 *     세 가지 이벤트(돌리는 중 / 회전 확정 / 보는 방향)를 낸다. 코치·음성은 그대로 쓴다.
 */

import type { ColorMove, State, View } from "../coach/cube.ts";
import { SegModel } from "./seg.ts";
import type { Event as TrackEvent, Roi } from "./tracker.ts";
import { Tracker, UNKNOWN } from "./tracker.ts";
import { FaceScanner, followRoi, NEED, readFace } from "./facescan.ts";
import { PoseSmoother } from "./smooth.ts";
import { hintArc, inLayer, stickers, stickyView } from "./hint.ts";
import type { Pose } from "./hint.ts";

const FACE_NAME = ["흰 면", "빨간 면", "초록 면", "노란 면", "주황 면", "파란 면"]; // 가운데 색 0~5

export interface VisionEvents {
  status(text: string): void;
  scanProgress(perSlot: number[], done: number): void;
  scanned(state: State): void;
  turning(color: number, angle: number): void;
  turnCancel(): void;
  /** 회전 확정. sync는 추적기 쪽 정보 (받는 쪽은 coach/resync.ts followMove 로 상태를 맞춘다) */
  move(m: ColorMove, sync?: MoveSync): void;
  pose(v: View): void;
  /** 추적 상태가 바뀔 때만 (idle / loading / scanning / searching / tracking) */
  trackState?(s: TrackState): void;
}

/**
 * 회전 확정과 함께 주는 추적기 정보.
 * 추적기는 놓친 회전을 뒤늦게 찾으면 몇 수 앞에 끼워 넣는다 (tracker.ts decide). 회전은 순서를 바꾸면 결과가 달라서,
 * 받는 쪽이 회전을 지금 상태 위에 얹기만 하면 추적기와 어긋난다. 그래서 추적기 상태를 함께 준다.
 */
export interface MoveSync {
  /** 몇 수 앞에 끼워 넣었나 (0 = 방금 돌린 회전) */
  back: number;
  /**
   * 이 회전까지 반영한 추적기 상태 (54칸 복사본). 한 프레임에 회전이 여럿이면 프레임이 끝난 뒤의 상태라
   * 마지막 회전에만 붙인다 (앱 설정에서는 한 프레임에 회전이 하나뿐이다: 의심 판정은 회전 확정 직후 세 프레임 안에는 안 나온다).
   */
  state?: State;
}

/** 화면에 보여 주는 추적 상태 */
export type TrackState = "idle" | "loading" | "scanning" | "searching" | "tracking";

/** 다음에 돌릴 층 (가운데 색, 시계 방향 90도 단위. 3이면 반시계 한 번) */
export interface LayerHint {
  color: number;
  k: 1 | 2 | 3;
}

const F = 0.85; // 초점거리 / 화면 너비 (노트북·휴대폰 카메라 화각 약 60~65도)
const HEX = ["#f4f4f0", "#d6233a", "#04a846", "#e9e02c", "#fa6a2c", "#0464bc"];
/** 잠깐(이 시간보다 짧게) 놓쳤다 다시 잡으면 '찾는 중'으로 바꾸지 않는다 (화면 배너 깜빡임 방지) */
const SEARCHING_AFTER_MS = 300;
/** 보는 방향(pose) 이벤트: 같은 방향이 이만큼 이어지고, 자세가 이만큼 잘 맞을 때만 낸다 */
const POSE_STABLE_FRAMES = 8, POSE_MIN_QUALITY = 0.6;

/** 프레임마다 재는 시간 (진단용) */
export interface FrameStat {
  t: number;
  modelMs: number;
  trackMs: number;
  drawMs: number;
  /** 자세를 넓게 다시 찾은 프레임이면 그 시간 */
  initMs: number;
  /** 이 프레임에 한 일: 면 읽기 / 자세 찾기(놓친 상태) / 추적 */
  kind: "scan" | "init" | "track";
  quality: number;
  /** 이 프레임을 처리하기 전까지 카메라·영상이 내보낸 프레임 수 (건너뛴 프레임 세기용) */
  presented: number;
}

/**
 * 관심영역이 화면 밖으로 거의 다 나가지 않게 당긴다 (aspect = 높이/너비).
 * 큐브를 화면 가장자리로 옮기면 영역 중심이 화면 밖으로 나가 잘라 넣을 그림이 거의 없어진다.
 * 영역의 3분의 1은 화면 안에 남게 중심을 제한한다. 크기는 화면을 많이 넘지 않게만 줄인다.
 */
export function clampRoi(roi: Roi, aspect: number): Roi {
  const side = Math.min(roi[2], 1.6);
  const mx = 0.5 + side / 2 - Math.min(side, 1) / 3;
  const my = aspect / 2 + side / 2 - Math.min(side, aspect) / 3;
  return [Math.min(Math.max(roi[0], -mx), mx), Math.min(Math.max(roi[1], -my), my), side];
}

export class VisionInput {
  phase: "idle" | "loading" | "scan" | "track" = "idle";
  fps = 0;
  /** UI용 추적 상태 */
  trackState: TrackState = "idle";
  /** UI용 일치율 0~1 (자세를 못 잡았으면 0) */
  quality = 0;
  private model: SegModel | null = null;
  private video: HTMLVideoElement;
  private view: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private stream: MediaStream | null = null;
  private tracker: Tracker | null = null;
  private scanner = new FaceScanner();
  private scanRoi: Roi | null = null;
  private lastSolveAt = 0;
  private busy = false;
  private running = false;
  private lastView = "";
  private viewStable = 0;
  private wasTurning = false;
  private frames = 0;
  private lostSince = 0;
  /** 화면을 거울처럼 보여 줄지 (카메라는 그렇게, 영상 파일은 그대로) — 보여 주기만 바뀐다 */
  private selfieView = false;
  /**
   * 카메라가 좌우반전된 화면을 주는가. 처음 상태 읽기에서 거울상으로 읽어야만 실제 큐브가 되면 true로 바꾼다.
   * 이후 모델에는 되돌린 화면을 넣는다 (모델·추적기는 늘 실제 세계와 같은 방향의 화면을 본다).
   */
  flipInput = false;
  private times: number[] = [];
  /** 다음에 돌릴 층 강조 (UI가 setHint 로 넣는다) */
  private hint: LayerHint | null = null;
  /** 최근 프레임 측정값 (최대 2000개, 진단·시험용) */
  stats: FrameStat[] = [];
  /** 바쁜 사이에 건너뛴 프레임 수 (진단용) */
  skipped = 0;
  /**
   * 큐브를 옮길 때 빨리 따라가기 (tracker.ts FollowOpts 전부). 실제 영상의 가만히 돌리는 구간(B)에서는
   * 한 번도 켜지지 않아 결과가 예전과 같고, 빠르게 옮길 때 추적 유지·다시 잡기가 크게 좋아진다. 끄면 예전 흐름.
   */
  fastFollow = true;
  private presented = 0;
  /** 화면에 그리는 자세 (추적기 자세를 부드럽게 거른 것) */
  private smoother = new PoseSmoother();
  private shownPose: Pose | null = null;
  private lastLayer = -1;
  private lastRoi: Roi | null = null;
  private everTracked = false;
  /** 보는 방향 판단용 (위 면·사람 쪽 면의 자리, 히스테리시스) */
  private viewSlots: { up: number; front: number } | null = null;
  private lastPoseSent = "";
  /** 낸 보는 방향 이벤트 수 (진단용) */
  poseEvents = 0;
  /**
   * 시작·멈춤 세대 번호. stop()마다 1씩 늘린다. 시작은 기다릴 때마다(모델 불러오기·카메라 권한·재생) 이 값이
   * 그대로인지 보고, 바뀌었으면(그사이 멈췄거나 다른 입력으로 바꿈) 조용히 그만둔다. 프레임 콜백과 모델 결과도 같다.
   */
  private gen = 0;
  /** 예약해 둔 다음 프레임 콜백 (멈추거나 다시 시작할 때 취소한다) */
  private frameReq: { id: number; vfc: boolean } | null = null;
  /** 불러오는 중인 모델 (그사이 다시 시작해도 한 번만 불러온다) */
  private modelLoad: Promise<SegModel> | null = null;

  private ev: VisionEvents;
  private base: string;

  // 매개변수 속성(private ev …)은 쓰지 않는다: node 가 타입만 지우고 바로 돌릴 수 있어야 시험에서 불러올 수 있다
  constructor(view: HTMLCanvasElement, ev: VisionEvents, base = "./") {
    this.ev = ev;
    this.base = base;
    this.view = view;
    this.ctx = view.getContext("2d")!;
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    // 영상이 끝나면 requestVideoFrameCallback 이 더 불리지 않으므로 끝 처리는 이 이벤트로 한다
    this.video.addEventListener("ended", () => this.onEnded());
    // 진단용: 개발자 도구나 시험 스크립트에서 바로 들여다볼 수 있게
    (globalThis as unknown as { __vision?: VisionInput }).__vision = this;
  }

  /**
   * 다음에 돌릴 층을 카메라 화면의 실제 큐브 위에 강조한다 (null이면 지움).
   * 추적 중인 자세로 투영하므로 큐브를 움직여도 따라간다.
   */
  setHint(m: LayerHint | null): void {
    this.hint = m && m.color >= 0 && m.color < 6 && [1, 2, 3].includes(m.k) ? { color: m.color, k: m.k } : null;
  }

  /** 영상 파일이 끝났을 때: 처리 고리를 멈추고 알린다 (카메라는 끝나지 않는다) */
  private onEnded(): void {
    if (!this.running) return;
    this.running = false;
    this.cancelFrame();
    this.ev.status("영상이 끝났어요.");
    this.setTrackState("idle");
  }

  private setTrackState(s: TrackState): void {
    if (s === this.trackState) return;
    this.trackState = s;
    this.ev.trackState?.(s);
  }

  private async ensureModel(): Promise<void> {
    if (this.model) return;
    this.phase = "loading";
    this.setTrackState("loading");
    this.ev.status("인식 모델을 불러오는 중…");
    // 런타임 파일 경로는 모듈 위치가 아니라 페이지 기준으로 풀어야 한다 (개발 서버·배포 모두)
    const abs = (p: string) => new URL(`${this.base}${p}`, document.baseURI).href;
    this.modelLoad ??= SegModel.load(abs("models/cube_seg_net.onnx"), abs("ort/"));
    try {
      this.model = await this.modelLoad;
    } catch (e) {
      this.modelLoad = null; // 다음 시작에서 다시 불러 본다
      throw e;
    }
  }

  /**
   * 시작 공통: 지금 것을 멈추고(세대 번호가 바뀐다) 새 번호를 받은 뒤, 기다릴 때마다 번호가 그대로인지 본다.
   * 그사이 멈췄거나 다른 입력으로 바꿨으면 옛 시작은 오류 없이 그만둔다 (main.ts 가 오류 안내를 띄우지 않게).
   * 멈춤에 끊긴 play()의 AbortError 같은 옛 시작의 오류도 알리지 않는다.
   */
  private async open(prepare: (stale: () => boolean) => Promise<void>): Promise<void> {
    this.stop();
    const g = this.gen;
    const stale = () => g !== this.gen;
    try {
      await this.ensureModel();
      if (stale()) return;
      await prepare(stale);
      if (stale()) return;
      this.begin();
    } catch (e) {
      if (!stale()) throw e;
    }
  }

  /** 카메라로 시작 */
  startCamera(): Promise<void> {
    return this.open(async (stale) => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" },
        audio: false,
      });
      if (stale()) {
        // 권한을 묻는 사이 멈췄거나 다른 입력으로 바꿨다: 늦게 열린 카메라는 바로 끈다
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      this.selfieView = true; // 거울처럼 보여 준다 (추적은 실제 방향의 화면으로)
      await this.video.play();
    });
  }

  /** 영상 파일로 시작 (카메라·큐브가 없을 때 시연용) */
  startFile(file: File): Promise<void> {
    return this.open(async () => {
      this.video.srcObject = null;
      this.video.src = URL.createObjectURL(file);
      this.video.loop = false;
      this.selfieView = false;
      await this.video.play();
    });
  }

  /** 주소로 영상 열기 (데모에 넣어 둔 샘플 영상) */
  startUrl(url: string): Promise<void> {
    return this.open(async () => {
      this.video.srcObject = null;
      this.video.src = url;
      this.video.loop = false;
      this.selfieView = false;
      await this.video.play();
    });
  }

  /** 상태를 이미 알 때(예: 다 맞춘 큐브) 스캔을 건너뛴다 */
  skipScan(state: State): void {
    this.toTrack(state);
  }

  rescan(): void {
    this.scanner = new FaceScanner();
    this.scanRoi = null;
    this.tracker = null;
    this.flipInput = false; // 다시 읽을 때마다 반전 여부도 다시 판단한다
    this.phase = "scan";
    this.quality = 0;
    this.smoother.reset();
    this.shownPose = null;
    this.everTracked = false;
    this.viewSlots = null;
    this.lastPoseSent = "";
    this.viewStable = 0;
    this.setTrackState("scanning");
    this.ev.status("돌리지 말고, 여섯 면을 한 면씩 카메라 정면으로 1초씩 보여 주세요. 방향은 상관없어요.");
  }

  /** 멈춤. 세대 번호를 바꿔, 기다리던 시작·예약된 프레임·처리 중인 모델 결과를 모두 버리게 한다. */
  stop(): void {
    this.gen++;
    this.running = false;
    this.cancelFrame();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.pause();
    this.phase = "idle";
    this.quality = 0;
    this.setTrackState("idle");
  }

  private begin(): void {
    this.view.width = this.video.videoWidth;
    this.view.height = this.video.videoHeight;
    this.rescan();
    this.running = true;
    this.cancelFrame(); // 고리는 늘 하나만
    this.loop(this.gen);
  }

  /** 예약해 둔 다음 프레임 콜백을 취소한다 (안 하면 멈췄다 다시 시작할 때 옛 콜백이 고리를 하나 더 만든다) */
  private cancelFrame(): void {
    const r = this.frameReq;
    if (!r) return;
    this.frameReq = null;
    const v = this.video as HTMLVideoElement & { cancelVideoFrameCallback?: (id: number) => void };
    if (r.vfc) v.cancelVideoFrameCallback?.(r.id);
    else cancelAnimationFrame(r.id);
  }

  private loop(g: number): void {
    if (!this.running || g !== this.gen) return;
    type Meta = { presentedFrames?: number };
    const next = (_t?: number, meta?: Meta) => {
      this.frameReq = null;
      if (g !== this.gen) return; // 취소하지 못한 옛 콜백 (멈춘 뒤 다시 시작했다)
      if (meta?.presentedFrames !== undefined) this.presented = meta.presentedFrames;
      this.loop(g);
    };
    const v = this.video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: (t: number, m: Meta) => void) => number };
    if (v.ended) {
      this.onEnded();
      return;
    }
    if (this.busy) {
      // 처리 중이어도 영상은 새 프레임으로 그린다 (전체 탐색처럼 오래 걸리는 프레임에서 화면이 멈춰 보이지 않게)
      this.skipped++;
      if (this.lastRoi) this.draw(this.tracker, this.lastRoi);
    } else void this.step();
    this.frameReq = v.requestVideoFrameCallback
      ? { id: v.requestVideoFrameCallback(next), vfc: true }
      : { id: requestAnimationFrame(() => next()), vfc: false };
  }

  private async step(): Promise<void> {
    const model = this.model;
    if (!model || this.phase === "idle" || this.phase === "loading") return;
    const g = this.gen;
    this.busy = true;
    try {
      const W = this.video.videoWidth, H = this.video.videoHeight;
      const tr = this.tracker;
      // 놓쳤을 때 찾을 영역: 추적 단계면 추적기가 정한다 (가운데 정사각형, 빨리 따라가기면 전체 너비와 번갈아),
      // 스캔 중이면 가운데 정사각형
      const lost: Roi = this.phase === "track" && tr ? tr.lostRoi(H / W) : [0, 0, H / W];
      const raw: Roi = (this.phase === "scan" ? this.scanRoi : tr?.roi()) ?? lost;
      const roi = clampRoi(raw, H / W);
      const t0 = performance.now();
      const maps = await model.run(this.video, W, H, roi, this.flipInput);
      // 기다리는 사이 멈췄거나 다시 시작했다: 옛 프레임의 결과로 새 스캔·추적을 건드리지 않는다
      // (busy 는 finally 에서 푼다. 멈출 때 바로 풀면 모델 실행 두 개가 겹칠 수 있다)
      if (g !== this.gen) return;
      const t1 = performance.now();
      const wasLost = !tr?.R;
      const events = tr && this.phase === "track" ? tr.update(maps, roi) : [];
      const t2 = performance.now();
      if (this.phase === "scan") {
        this.scanRoi = followRoi(maps, roi);
        const face = readFace(maps);
        if (face) this.scanner.add(face);
      }
      this.frames++;
      const now = performance.now();
      this.times.push(now);
      while (this.times.length && now - this.times[0] > 1000) this.times.shift();
      this.fps = this.times.length;
      this.updateShown(tr, wasLost, events.length > 0, now);
      if (this.phase === "scan") this.scanStep();
      else if (this.phase === "track" && tr) this.trackStep(tr, events);
      this.lastRoi = roi;
      this.draw(this.tracker, roi);
      const t3 = performance.now();
      const kind = this.phase === "scan" || !tr ? "scan" : wasLost ? "init" : "track";
      this.stats.push({
        t: Math.round(t0), modelMs: t1 - t0, trackMs: t2 - t1, drawMs: t3 - t2,
        initMs: kind === "init" ? t2 - t1 : 0, kind, quality: this.quality, presented: this.presented,
      });
      if (this.stats.length > 2000) this.stats.shift();
    } catch (e) {
      if (g === this.gen) this.ev.status(`처리 중 오류: ${(e as Error).message}`);
    } finally {
      this.busy = false;
    }
  }

  private scanStep(): void {
    // 면마다 같은 격자가 NEED번 읽히면 다 찬 것으로 보여 준다 (막대 0~8칸 눈금에 맞춤).
    // 가운데 색만 맞는 틀린 읽기로는 차지 않으므로, 덜 읽힌 면이 화면에 그대로 남는다.
    const ag = [0, 1, 2, 3, 4, 5].map((c) => this.scanner.agree(c));
    this.ev.scanProgress(ag.map((a) => Math.round((Math.min(a, NEED) / NEED) * 8)), ag.filter((a) => a >= NEED).length * 8);
    if (!this.scanner.ready() || performance.now() - this.lastSolveAt < 500) return;
    this.lastSolveAt = performance.now();
    const s = this.scanner.solve();
    if (s) {
      if (s.mirrored) this.flipInput = true; // 이 카메라는 좌우반전된 화면을 준다
      this.toTrack(s.state);
    }
    else if (this.scanner.ambiguous)
      // 다 맞춘 상태에 가까운 큐브는 여섯 면이 서로 돌린 모양이라 면만으로는 가릴 수 없다 (web/src/vision/facescan.ts)
      this.ev.status("여섯 면을 다 읽었지만, 면만 봐서는 큐브 상태가 두 가지 이상으로 읽혀요. 다 맞춘 상태에 가까우면 생길 수 있어요. 아무 층이나 몇 번 더 돌려 섞은 뒤 '다시 읽기'를 눌러 주세요.");
    else this.ev.status(`여섯 면이 서로 맞지 않아요. ${FACE_NAME[this.scanner.weakest()]}을 한 번 더 정면으로 보여 주세요.`);
  }

  private toTrack(s: State): void {
    this.tracker = new Tracker(s, F, 2, 1500); // 자세는 다음 프레임에 넓게 찾는다 (상태를 다 알면 잘 잡힌다)
    this.tracker.fastFollow = this.fastFollow; // 큐브를 옮기면 바로 따라가기 (web/src/vision/tracker.ts)
    this.phase = "track";
    this.setTrackState("searching");
    this.ev.scanned(s);
    if (this.flipInput) this.ev.status("다 읽었어요. 카메라 화면이 좌우반전돼 있어서 되돌려서 따라갑니다.");
  }

  private trackStep(tr: Tracker, events: TrackEvent[]): void {
    events.forEach((e, i) => {
      const state = i === events.length - 1 ? Uint8Array.from(tr.colors) : undefined;
      this.ev.move({ color: e.center, k: e.quarterTurns as 1 | 2 | 3 }, { back: e.info?.back ?? 0, state });
    });
    const t = tr.turning();
    if (t) this.ev.turning(t.color, t.angle);
    else if (this.wasTurning && !events.length) this.ev.turnCancel();
    this.wasTurning = !!t;
    if (!tr.R) {
      const now = performance.now();
      this.lostSince ||= now;
      // 잠깐 놓친 동안(SEARCHING_AFTER_MS)은 '따라가는 중'과 직전 일치율을 그대로 둔다 (막대·배너 깜빡임 방지)
      if (!this.everTracked || now - this.lostSince > SEARCHING_AFTER_MS) {
        this.setTrackState("searching");
        this.quality = 0;
      }
      if (now - this.lostSince > 1500) this.ev.status("큐브를 다시 찾는 중이에요. 카메라 앞에 보여 주세요.");
      return;
    }
    // 일치율은 막대가 덜 떨리게 조금 눌러서 보여 준다
    this.quality = this.quality > 0 ? 0.6 * this.quality + 0.4 * tr.quality : tr.quality;
    this.everTracked = true;
    this.setTrackState("tracking");
    this.lostSince = 0;
    this.poseStep(tr, !!t);
  }

  /**
   * 보는 방향(pose) 이벤트. 코치의 기호 기준이 바뀌므로 사람이 실제로 통째로 돌려 잡았을 때만 내야 한다.
   *  - 표시용 평활 자세로 판단하고 (프레임마다의 흔들림 제거)
   *  - 큐브를 45도쯤 비스듬히 들어 두 면이 비길 때 왔다 갔다 하지 않게 히스테리시스를 두고 (hint.ts stickyView)
   *  - 층을 돌리는 중이거나 자세가 덜 맞을 때는 판단하지 않으며
   *  - 같은 방향이 POSE_STABLE_FRAMES 이어질 때, 마지막으로 낸 방향과 다를 때만 낸다.
   */
  private poseStep(tr: Tracker, turning: boolean): void {
    const P = this.shownPose;
    if (!P || turning || tr.layer >= 0 || tr.quality < POSE_MIN_QUALITY) return;
    this.viewSlots = stickyView(P.R, this.viewSlots);
    const v = { up: tr.colors[this.viewSlots.up * 9 + 4], front: tr.colors[this.viewSlots.front * 9 + 4] };
    const key = `${v.up},${v.front}`;
    this.viewStable = key === this.lastView ? this.viewStable + 1 : 0;
    this.lastView = key;
    if (this.viewStable >= POSE_STABLE_FRAMES && key !== this.lastPoseSent) {
      this.lastPoseSent = key;
      this.poseEvents++;
      this.ev.pose(v);
    }
  }

  /** 표시용 자세를 갱신한다 (다시 잡았을 때·층 회전이 바뀔 때·회전이 확정될 때는 거르지 않고 바로 옮긴다) */
  private updateShown(tr: Tracker | null, wasLost: boolean, moved: boolean, nowMs: number): void {
    if (!tr?.R || !tr.Tv || this.phase !== "track") {
      this.smoother.reset();
      this.shownPose = null;
      this.lastLayer = -1;
      return;
    }
    const snap = wasLost || moved || tr.layer !== this.lastLayer;
    this.smoother.update(tr.R, tr.Tv, nowMs / 1000, snap);
    this.lastLayer = tr.layer;
    this.shownPose = { R: this.smoother.R!, T: this.smoother.T!, layer: tr.layer, phi: tr.phi };
  }

  private draw(tr: Tracker | null, roi: Roi): void {
    const ctx = this.ctx, W = this.view.width, H = this.view.height;
    // 원본 → 실제 방향: flipInput이면 뒤집기. 실제 방향 → 보여 주기: selfieView면 뒤집기.
    ctx.save();
    if (this.flipInput !== this.selfieView) ctx.setTransform(-1, 0, 0, 1, W, 0);
    ctx.drawImage(this.video, 0, 0, W, H);
    ctx.restore();
    // 관심영역과 칸 표시는 실제 방향 좌표로 계산한다 (selfieView면 영상과 함께 뒤집힌다)
    ctx.save();
    if (this.selfieView) ctx.setTransform(-1, 0, 0, 1, W, 0);
    const lw = Math.max(2, W / 400);
    // 관심영역: 잘 따라갈 때는 옅게, 찾는 중에는 눈에 띄게
    const side = roi[2] * W, cx = W / 2 + roi[0] * W, cy = H / 2 + roi[1] * W;
    const pose = tr?.R ? this.shownPose : null;
    ctx.strokeStyle = pose ? "rgba(255,255,255,0.25)" : "rgba(255,200,0,0.8)";
    ctx.lineWidth = pose ? lw / 2 : lw;
    ctx.setLineDash([10, 8]);
    ctx.strokeRect(cx - side / 2, cy - side / 2, side, side);
    ctx.setLineDash([]);
    if (pose && tr) {
      // 추적기가 생각하는 칸 위치와 색 (평활한 자세로 그려 덜 떨린다)
      const sx = (u: number) => W / 2 + u * W, sy = (v: number) => H / 2 + v * W;
      const st = stickers(pose, F);
      const unitPx = (F / pose.T[2]) * W; // 조각 한 변의 화면 픽셀
      const r = Math.max(3, unitPx * 0.13);
      // 힌트 층 = 가운데 칸 색이 힌트 색인 면 (가운데 칸은 돌려도 그 자리에 있다)
      let L = -1;
      if (this.hint) for (let f = 0; f < 6; f++) if (tr.colors[f * 9 + 4] === this.hint.color) L = f;
      for (let k = 0; k < 54; k++) {
        if (st.w[k] <= 0.05) continue;
        const c = tr.colors[k];
        const x = sx(st.u[k]), y = sy(st.v[k]);
        const mark = L >= 0 && inLayer(L, k);
        ctx.globalAlpha = (mark ? 0.55 : 0.25) + (mark ? 0.45 : 0.55) * st.w[k];
        ctx.beginPath();
        ctx.arc(x, y, mark ? r * 1.35 : r, 0, Math.PI * 2);
        if (c === UNKNOWN) {
          ctx.strokeStyle = "rgba(255,255,255,0.8)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          ctx.fillStyle = HEX[c];
          ctx.fill();
          ctx.strokeStyle = mark ? "#fff" : "rgba(0,0,0,0.55)";
          ctx.lineWidth = mark ? Math.max(2, r * 0.45) : 1.5;
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      if (L >= 0 && this.hint) this.drawArrow(pose, L, this.hint.k, unitPx, sx, sy);
    }
    ctx.restore();
  }

  /** 다음에 돌릴 층의 회전 방향 화살표 (3차원 곡선을 투영한 것이라 영상과 함께 거울로 뒤집혀도 방향이 맞다) */
  private drawArrow(pose: Pose, L: number, k: 1 | 2 | 3, unitPx: number, sx: (u: number) => number, sy: (v: number) => number): void {
    const arc = hintArc(pose, F, L, k);
    if (!arc || arc.pts.length < 3) return;
    const ctx = this.ctx;
    const pts = arc.pts.map(([u, v]) => [sx(u), sy(v)] as [number, number]);
    const lw = Math.max(3, unitPx * 0.16);
    const head = (i: number) => {
      // i번째 점에서 진행 방향으로 머리
      const [x1, y1] = pts[i], [x0, y0] = pts[Math.max(0, i - 2)];
      const a = Math.atan2(y1 - y0, x1 - x0), hs = Math.max(12, unitPx * 0.6);
      const dx = Math.cos(a), dy = Math.sin(a);
      // 끝점에서 진행 방향으로 뾰족한 이등변 삼각형 (밑변이 곡선 끝에 붙는다)
      ctx.beginPath();
      ctx.moveTo(x1 + dx * hs * 0.9, y1 + dy * hs * 0.9);
      ctx.lineTo(x1 - dx * hs * 0.1 - dy * hs * 0.5, y1 - dy * hs * 0.1 + dx * hs * 0.5);
      ctx.lineTo(x1 - dx * hs * 0.1 + dy * hs * 0.5, y1 - dy * hs * 0.1 - dx * hs * 0.5);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    };
    for (const [color, width] of [["rgba(0,0,0,0.55)", lw + 4], ["#ffd23f", lw]] as [string, number][]) {
      ctx.strokeStyle = color;
      ctx.fillStyle = "#ffd23f";
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 2;
    head(pts.length - 1);
    if (arc.heads === 2) head(Math.floor(pts.length / 2));
  }
}

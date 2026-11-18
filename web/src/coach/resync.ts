/**
 * 카메라 추적기 상태 따라가기.
 *
 * 추적기는 놓쳤던 회전을 뒤늦게 찾으면 몇 수 앞에 끼워 넣고 상태를 다시 계산한다 (tracker.ts decide, back > 0).
 * 회전은 순서를 바꾸면 결과가 달라지므로, 받은 회전을 지금 상태 위에 얹기만 하면 추적기 상태와 어긋난다.
 * 그래서 카메라는 회전과 함께 추적기 상태를 주고(camera.ts MoveSync), 화면·코치는 그 상태를 따른다.
 * 키보드(가상 큐브)는 추적기 상태가 없으므로 예전처럼 위에 얹는다.
 *
 * 견주기 전에 큐브 방향을 맞춘다: 추적기는 '층을 잡고 몸통을 돌린' 회전을 그대로 기록해서(tracker.ts CAND_PERMS 의
 * 홀수 번째) 54칸 배열에서 가운데 칸 자리가 바뀔 수 있다. 같은 큐브를 통째로 돌려 본 것일 뿐이라, 코치 쪽 배열
 * (가운데 칸이 늘 그 자리)에 맞게 통째로 돌린 뒤 견준다.
 */

import type { ColorMove, State } from "./cube.ts";
import { applyColorMove, applySlotAlg } from "./cube.ts";

/** 큐브를 통째로 돌리는 24가지 (위로 올 면 6 x 그 둘레 4) */
export const WHOLE_TURNS: string[] = ["", "x", "x2", "x'", "z", "z'"].flatMap((a) =>
  ["", "y", "y2", "y'"].map((b) => [a, b].filter(Boolean).join(" ")),
);

export function sameState(a: State, b: State): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** s를 통째로 돌려 가운데 칸 여섯 개가 ref 와 같은 자리에 오게 한 상태 (맞는 방향이 없으면 s 그대로) */
export function alignCenters(s: State, ref: State): State {
  for (const turn of WHOLE_TURNS) {
    const r = turn ? applySlotAlg(s, turn) : s;
    let ok = true;
    for (let slot = 0; slot < 6 && ok; slot++) ok = r[slot * 9 + 4] === ref[slot * 9 + 4];
    if (ok) return r;
  }
  return s;
}

/**
 * 회전 하나를 받은 뒤의 상태.
 *  - tracked(추적기 상태)가 없거나, 위에 얹은 결과와 같으면: 위에 얹은 결과 (resync = false, 코치에는 onMove)
 *  - 다르면(놓친 회전을 앞에 끼워 넣음): 추적기 상태를 코치 쪽 방향으로 맞춘 것 (resync = true, 코치·연습은 그 상태로 다시 맞춘다)
 * back > 0 이어도 끼워 넣은 회전이 뒤 회전들과 순서를 바꿔도 같은 회전(마주 보는 면 등)이면 결과가 같아 그대로 간다.
 */
export function followMove(cube: State, m: ColorMove, tracked?: State | null): { state: State; resync: boolean } {
  const stacked = applyColorMove(cube, m);
  if (!tracked) return { state: stacked, resync: false };
  const t = alignCenters(tracked, stacked);
  if (sameState(stacked, t)) return { state: stacked, resync: false };
  return { state: Uint8Array.from(t), resync: true };
}

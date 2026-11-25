// Run with `npm test` (Node's built-in test runner; Node strips the TypeScript types).
import assert from "node:assert/strict";
import { test } from "node:test";

import { largestMoments } from "../src/moments.ts";

const times = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 3.0, 5.0, 7.0];

test("the first second is skipped even when it holds the largest change", () => {
  // The memory filling up from zero at the start gives the biggest bar at 0.2 s.
  const change = [0, 3.0, 2.5, 2.0, 1.5, 0.4, 0.3, 0.9, 0.2, 0.6];
  const moments = largestMoments(change, times);
  assert.ok(moments.every((i) => times[i] >= 1), `picked ${moments.map((i) => times[i])}`);
  assert.deepEqual(moments, [7, 9, 5]); // 3.0 s, 7.0 s, then 1.0 s, largest first
});

test("moments stay at least two seconds apart", () => {
  const change = [0, 0, 0, 0, 0, 0.9, 1.0, 0.8, 0.1, 0.1];
  // 1.2 s is largest; 1.0 s and 3.0 s are within 2 s of it, so 5.0 s and 7.0 s follow.
  assert.deepEqual(largestMoments(change, times), [6, 8, 9]);
});

test("the skip and the spacing can be changed, and nothing is named without a change", () => {
  const change = [0, 3.0, 0, 0, 0, 0, 0, 0.5, 0, 0];
  assert.deepEqual(largestMoments(change, times, { skipFirst: 0 }), [1, 7]);
  assert.deepEqual(largestMoments(change.map(() => 0), times), []);
});

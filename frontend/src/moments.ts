/**
 * The time columns to name in a "This round" note: the largest values, biggest first, at
 * least `apart` seconds from each other. Columns before `skipFirst` seconds are left out,
 * because the GRU's memory starts at zero and its first updates mostly reflect it filling
 * up, not anything in the round. The chart itself still shows those columns.
 */
export function largestMoments(
  values: number[],
  times: number[],
  { count = 3, apart = 2, skipFirst = 1 }: { count?: number; apart?: number; skipFirst?: number } = {},
): number[] {
  const order = values
    .map((_, i) => i)
    .filter((i) => values[i] > 0 && times[i] >= skipFirst)
    .sort((a, b) => values[b] - values[a]);
  const chosen: number[] = [];
  for (const i of order) {
    if (chosen.every((j) => Math.abs(times[i] - times[j]) >= apart)) chosen.push(i);
    if (chosen.length === count) break;
  }
  return chosen;
}

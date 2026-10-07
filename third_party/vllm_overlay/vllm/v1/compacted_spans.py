"""Per-KV-cache-group compaction bookkeeping for vllm.v1.request.Request (AbridgeKV overlay).

Pure Python (no vLLM imports) so the CPU tests can exercise it directly.
"""
import bisect


class CompactedSpanIndex:
    """Sorted non-overlapping removed spans of one KV cache group + O(log n) position maps."""

    __slots__ = ("spans", "_starts", "_cumulative", "_gap_physical", "_gap_logical", "total_removed")

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self.spans = spans
        starts: list[int] = []
        cumulative: list[int] = []
        gap_physical: list[int] = [0]
        gap_logical: list[int] = [0]
        total = 0
        for s, e in spans:
            starts.append(s)
            total += e - s
            cumulative.append(total)
            gap_physical.append(e - total)
            gap_logical.append(e)
        self._starts = starts
        self._cumulative = cumulative
        self._gap_physical = gap_physical
        self._gap_logical = gap_logical
        self.total_removed = total

    def is_token_active(self, position: int) -> bool:
        if not self._starts:
            return True
        idx = bisect.bisect_right(self._starts, position) - 1
        if idx < 0:
            return True
        return position >= self.spans[idx][1]

    def get_active_positions(self, num_tokens: int) -> list[int]:
        if not self.spans:
            return list(range(num_tokens))
        active: list[int] = []
        prev_end = 0
        for start, end in self.spans:
            if start >= num_tokens:
                break
            active.extend(range(prev_end, min(start, num_tokens)))
            prev_end = end
        if prev_end < num_tokens:
            active.extend(range(prev_end, num_tokens))
        return active

    def logical_to_physical(self, logical_pos: int) -> int:
        if not self._starts:
            return logical_pos
        idx = bisect.bisect_right(self._starts, logical_pos) - 1
        if idx < 0:
            return logical_pos
        _, end = self.spans[idx]
        if logical_pos >= end:
            return logical_pos - self._cumulative[idx]
        prev_cum = self._cumulative[idx - 1] if idx > 0 else 0
        return logical_pos - prev_cum - (logical_pos - self._starts[idx])

    def physical_to_logical(self, physical_pos: int) -> int:
        idx = bisect.bisect_right(self._gap_physical, physical_pos) - 1
        return self._gap_logical[idx] + (physical_pos - self._gap_physical[idx])

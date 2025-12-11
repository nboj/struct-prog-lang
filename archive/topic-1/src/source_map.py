from bisect import bisect_right
from src.types import Span


class SourceMap:
    def __init__(self, text: str):
        self.text = text
        self.line_starts = [0]
        for i, ch in enumerate(text):
            if ch == '\n':
                self.line_starts.append(i + 1)

    def offset_to_line_col(self, off: int) -> tuple[int, int]:
        # 1-based line/col
        li = bisect_right(self.line_starts, off) - 1
        return (li + 1, off - self.line_starts[li] + 1)

    def span_to_lc(self, span: Span):
        return (self.offset_to_line_col(span.start),
                self.offset_to_line_col(span.end))

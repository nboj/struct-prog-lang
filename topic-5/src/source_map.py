from bisect import bisect_right
from .types import Span
from typing import Any


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

    def err_here(self, tok: Any, msg: str) -> str:
        span = getattr(tok, "span", tok)
        (sline, scol), (eline, ecol) = self.span_to_lc(span)
        lines = self.text.splitlines(keepends=False)

        def gutter_at(line: int):
            return " | " + str(line+1) + " "
        out = msg + "\n"
        if sline - 2 >= 0:
            out += gutter_at(sline-2) + lines[sline-2] + "\n"
        if sline - 1 >= 0:
            out += gutter_at(sline-1) + lines[sline-1] + "\n"
            out += "".rjust((scol-1)+len(gutter_at(sline-1)), " ")
            out += "".rjust(ecol-scol, "^")
        return out

    def to_err(self, node: Any, msg: str) -> str:
        span = getattr(node, "span", node)

        (sline, scol), (eline, ecol) = self.span_to_lc(span)
        lines = self.text.splitlines(keepends=False)

        def get_line(ln: int) -> str:
            return lines[ln - 1] if 1 <= ln <= len(lines) else ""

        parts: list[str] = []
        header = f"At {sline}:{scol}" + (f"-{eline}:{ecol}" if (sline, scol) != (eline, ecol) else "")
        parts.append(header)

        if sline == eline:
            line = get_line(sline)
            parts.append(f" {sline:>4} | {line}")
            width = max(1, ecol - scol)
            caret = " " * (scol - 1) + "^" * width + f" {msg}"
            parts.append("      | " + caret)
        else:
            first = get_line(sline)
            parts.append(f" {sline:>4} | {first}")
            first_carets = " " * (scol - 1) + "^" * \
                max(1, max(0, len(first) - (scol - 1)))
            parts.append("      | " + first_carets)

            if eline - sline > 1:
                parts.append("      | ...")

            last = get_line(eline)
            parts.append(f" {eline:>4} | {last}")
            last_carets = "^" * max(1, min(max(0, ecol - 1), len(last)))
            parts.append("      | " + last_carets + f" {msg}")

        return "\n".join(parts)

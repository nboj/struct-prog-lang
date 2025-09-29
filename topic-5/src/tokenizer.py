from enum import Enum
from .types import Span
from .source_map import SourceMap
from typing import override


class TokenType(Enum):
    Number = "Number"
    Plus = "Plus"
    Minus = "Minus"
    Star = "Star"
    Amp = "Amp"
    DoubleAmp = "DoubleAmp"
    DoublePipe = "DoublePipe"
    Ident = "Ident"
    Nl = "Newline"
    Eq = "Equals"
    NEq = "NotEquals"
    DbEq = "DoubleEquals"
    Divide = "Divide"
    OpenParen = "OpenParen"
    CloseParen = "CloseParen"
    OpenCurly = "OpenCurly"
    CloseCurly = "CloseCurly"
    Semi = "SemiColon"
    Bang = "Bang"
    Comma = "Comma"
    Dot = "Dot"
    Eof = "Eof"
    String = "String"
    Gt = "GreaterThan"
    Lt = "LessThan"

    # KeyWords
    Let = "Let"
    If = "If"
    Else = "Else"
    While = "While"
    Loop = "Loop"
    BTrue = "BooleanTrue"
    BFalse = "BooleanFalse"


class Token:
    kind: TokenType
    span: Span
    raw: str

    def __init__(self, kind: TokenType, raw: str, span: Span):
        self.kind = kind
        self.raw = raw
        self.span = span

    @override
    def __repr__(self):
        return f"{self.raw}"


class Tokenizer:
    tokens: list[Token]
    text: str
    index: int
    sm: SourceMap
    KEYWORDS: dict[str, TokenType] = {
        "let": TokenType.Let,
        "if": TokenType.If,
        "else": TokenType.Else,
        "true": TokenType.BTrue,
        "false": TokenType.BFalse,
        "while": TokenType.While,
    }

    def __init__(self, text: str):
        self.text = text
        self.sm = SourceMap(text)
        self.tokens = []
        self.index = 0

    def is_num(self, ch: str):
        return ch >= "0" and ch <= "9"

    def is_alpha(self, ch: str):
        return (ch >= "A" and ch <= "Z") or (ch >= "a" and ch <= "z") or ch == "_"

    def is_alphanum(self, ch: str):
        return self.is_alpha(ch) or self.is_num(ch)

    def bump(self) -> str:
        tmp = self.peek()
        self.index += 1
        return tmp

    def peek(self) -> str:
        assert self.index < len(self.text), "tried peeking after last token"
        return self.text[self.index]

    def peek_next(self) -> str:
        assert self.index < len(self.text) - 1, "tried peeking after last token"
        if self.index + 1 >= len(self.text):
            return "\0"
        return self.text[self.index + 1]

    def add(self, kind: TokenType, raw: str, span: Span):
        self.tokens.append(Token(kind, raw, span))

    def add_single_at(self, kind: TokenType, raw: str, index: int):
        self.tokens.append(Token(kind, raw, Span(index, index + 1)))

    def add_single(self, kind: TokenType, raw: str):
        self.tokens.append(Token(kind, raw, Span(self.index, self.index + 1)))

    def tokenize(self) -> list[Token]:
        self.tokens = []
        self.index = 0
        assert len(self.text) > 0, "Err: Invalid Input."
        while self.index < len(self.text):
            ch = self.peek()
            if ch in (" ", "\r", "\t"):
                _ = self.bump()
                continue

            if ch == "\n":
                self.add_single(TokenType.Nl, ch)
                _ = self.bump()
                continue

            if ch == "=":
                index = self.index
                _ = self.bump()
                if self.peek() == "=":
                    _ = self.bump()
                    self.add(TokenType.DbEq, ch + ch, Span(index, self.index))
                else:
                    self.add(TokenType.Eq, ch, Span(index, self.index))
                continue

            if ch == "-":
                self.add_single(TokenType.Minus, ch)
                _ = self.bump()
                continue

            if ch == ".":  # '.0)'
                if self.is_num(self.peek_next()):
                    start_on = self.index
                    out = "0" + self.bump()
                    ch = self.bump()
                    out += ch
                    ch = self.peek()
                    while self.is_num(ch):
                        out += self.bump()
                        ch = self.peek()
                    self.add(TokenType.Number, out, Span(start_on, self.index))
                else:
                    self.add_single(TokenType.Dot, ch)
                    _ = self.bump()
                continue

            if ch == "+":
                self.add_single(TokenType.Plus, ch)
                _ = self.bump()
                continue

            if ch == ",":
                self.add_single(TokenType.Comma, ch)
                _ = self.bump()
                continue

            if ch == "(":
                self.add_single(TokenType.OpenParen, ch)
                _ = self.bump()
                continue

            if ch == ")":
                self.add_single(TokenType.CloseParen, ch)
                _ = self.bump()
                continue

            if ch == "{":
                self.add_single(TokenType.OpenCurly, ch)
                _ = self.bump()
                continue

            if ch == "}":
                self.add_single(TokenType.CloseCurly, ch)
                _ = self.bump()
                continue

            if ch == ">":
                self.add_single(TokenType.Gt, ch)
                _ = self.bump()
                continue

            if ch == "<":
                self.add_single(TokenType.Lt, ch)
                _ = self.bump()
                continue

            if ch == ";":
                self.add_single(TokenType.Semi, ch)
                _ = self.bump()
                continue

            if ch == "*":
                self.add_single(TokenType.Star, ch)
                _ = self.bump()
                continue
            if ch == "|":
                if self.peek_next() == "|":
                    start_index = self.index
                    _ = self.bump()
                    _ = self.bump()
                    self.add(
                        TokenType.DoublePipe,
                        raw="||",
                        span=Span(start_index, self.index),
                    )
                    continue
                else:
                    raise AssertionError(
                        self.err_at(
                            self.index,
                            f"Unknown character next to pipe: {self.peek_next()}",
                        )
                    )

            if ch == "&":
                start_index = self.index
                _ = self.bump()
                if self.peek() == "&":
                    _ = self.bump()
                    self.add(TokenType.DoubleAmp, "&&", Span(start_index, self.index))
                else:
                    self.add(TokenType.Amp, "&", Span(start_index, self.index))
                continue

            if ch == "!":
                start_index = self.index
                _ = self.bump()
                if self.peek() == "=":
                    _ = self.bump()
                    self.add(TokenType.NEq, "!=", Span(start_index, self.index))
                else:
                    self.add(TokenType.Bang, "!", Span(start_index, self.index))
                continue

            if ch == "/":
                next = self.peek_next()
                if next == "/":
                    while self.peek() not in ("\n", "\0"):
                        _ = self.bump()
                    continue
                elif next == "*":
                    start_on = self.index
                    _ = self.bump()  # /
                    _ = self.bump()  # *
                    next = self.peek()
                    while (
                        not (next == "*" and self.peek_next() == "/") and next != "\0"
                    ):
                        _ = self.bump()
                        next = self.peek()
                    if self.peek() == "\0":
                        raise AssertionError(
                            self.err_at(start_on, "Unterminated block comment")
                        )
                    _ = self.bump()  # *
                    _ = self.bump()  # /
                    continue
                else:
                    self.add_single(TokenType.Divide, "/")
                    _ = self.bump()
                    continue

            if ch == '"':
                start = self.index
                out = ""
                _ = self.bump()
                ch = self.peek()
                while ch != '"':
                    out += ch
                    if ch == "\0" or ch == "\n":
                        raise AssertionError(
                            self.err_at(self.index, "unclosed string literal")
                        )
                    _ = self.bump()
                    ch = self.peek()
                _ = self.bump()
                self.add(TokenType.String, out, Span(start, self.index))
                continue

            if self.is_num(ch):
                num = ""
                peeked = ch
                start_index = self.index
                found_dot = False
                while self.is_num(peeked) or (peeked == "." and not found_dot):
                    if peeked == ".":
                        found_dot = True
                    num += self.bump()
                    peeked = self.peek()
                self.add(TokenType.Number, num, Span(start_index, self.index))
                continue

            if self.is_alpha(ch):
                ident = ""
                peeked = ch
                start_index = self.index
                while self.is_alphanum(peeked):
                    ident += self.bump()
                    peeked = self.peek()
                keyword = self.KEYWORDS.get(ident)
                self.add(
                    keyword if keyword is not None else TokenType.Ident,
                    ident,
                    Span(start_index, self.index),
                )
                continue

            raise AssertionError(self.err_at(self.index, f"Unknown character {ch}"))

        self.add_single(TokenType.Eof, "EOF")
        return self.tokens

    def err_at(self, off: int, msg: str) -> str:
        (line, col) = self.sm.offset_to_line_col(off)
        src_lines = self.text.splitlines()
        src = src_lines[line - 1] if 1 <= line <= len(src_lines) else ""
        caret = " " * (col - 1) + "^ " + msg
        return f"At {line}:{col}\n {line:>4} | {src}\n       | {caret}"

    def debug_print(self):
        print(self.text)
        assert len(self.tokens) > 0, "Cannot debug print with empty token array."
        largest_tok = max(self.tokens, key=lambda t: len(t.raw))
        target_len = 8
        final = "[\n"
        indent_level = 1
        for token in self.tokens:
            if token.kind == TokenType.CloseCurly:
                indent_level -= 1
            if token.kind == TokenType.Nl:
                final += "\n"
                continue
            final += "\t" * indent_level
            final += f"{token.raw}"
            pad = len(largest_tok.raw) - len(token.raw)
            extra = 0
            if (len(token.raw) + pad) < target_len:
                extra = target_len - (len(token.raw) + pad)
            final += " " * (pad + extra)
            final += f":{token.kind.name}, \n"
            if token.kind == TokenType.OpenCurly:
                indent_level += 1
        if len(final) >= 2:
            final = final[:-2]

        final += "\n]"
        print(final)

    def _escape(self, s: str) -> str:
        return s.encode("unicode_escape").decode("ascii")

    def tokens_debug(self) -> str:
        out = []
        for t in self.tokens:
            (sline, scol), (eline, ecol) = self.sm.span_to_lc(t.span)
            out.append(
                f'Token {{ kind: {t.kind.name}, raw: "{self._escape(t.raw)}", '
                f"span: [{t.span.start},{
                    t.span.end}) @ {sline}:{scol}-{eline}:{ecol} }}"
            )
        return "\n".join(out)

    def tokens_pretty_gutter(self) -> str:
        lines = self.text.splitlines(keepends=False)
        pieces = []
        for ln, line in enumerate(lines, start=1):
            line_start = self.sm.line_starts[ln - 1]
            line_end = (
                self.sm.line_starts[ln]
                if ln < len(self.sm.line_starts)
                else len(self.text)
            )
            carets = [" "] * len(line)
            for t in self.tokens:
                s = max(t.span.start, line_start)
                e = min(t.span.end, line_end)
                if s < e:
                    s_col = s - line_start
                    # at least one caret
                    e_col = max(s_col + 1, e - line_start)
                    for i in range(s_col, min(e_col, len(line))):
                        carets[i] = "^"
            pieces.append(f"{ln:>4} | {line}")
            if any(c != " " for c in carets):
                pieces.append("     | " + "".join(carets))
        pieces.append("")
        pieces.append(self.tokens_debug())
        return "\n".join(pieces)

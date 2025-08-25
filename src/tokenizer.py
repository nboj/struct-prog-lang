from enum import Enum


class TokenType(Enum):
    Number = "Number"
    Plus = "Plus"
    Minus = "Minus"
    Star = "Star"
    Amp = "Amp"
    DoubleAmp = "DoubleAmp"
    Ident = "Ident"
    Nl = "Newline"
    Eq = "Equals"
    DbEq = "DoubleEquals"
    Divide = "Divide"
    LParen = "LParen"
    RParen = "RParen"
    LCurly = "LCurly"
    RCurly = "RCurly"
    Dot = "Dot"
    Eof = "Eof"

    # KeyWords
    Let = "Let"
    If = "If"


class Token:
    kind: TokenType
    row: int
    col: int
    raw: str

    def __init__(self, kind: TokenType, raw: str, row: int, col: int):
        self.kind = kind
        self.raw = raw
        self.row = row
        self.col = col

    def __repr__(self):
        return f"{self.raw}"


class Tokenizer:
    tokens: list[Token]
    text: str
    row: int
    col: int
    index: int
    KEYWORDS: dict

    def __init__(self, text: str):
        self.text = text
        self.KEYWORDS = {
            "let": TokenType.Let,
            "if": TokenType.If,
        }

    def is_num(self, ch: str):
        return (ch >= '0' and ch <= '9')

    def is_alpha(self, ch: str):
        return (ch >= 'A' and ch <= 'Z') or (ch >= 'a' and ch <= 'z') or ch == '_'

    def is_alphanum(self, ch: str):
        return self.is_alpha(ch) or self.is_num(ch)

    def bump(self) -> str:
        tmp = self.peek()
        self.index += 1
        if tmp == '\n':
            self.row += 1
            self.col = 1
        else:
            self.col += 1
        return tmp

    def peek(self) -> str:
        if self.index >= len(self.text):
            return "\0"
        return self.text[self.index]

    def peek_next(self) -> str:
        if self.index+1 >= len(self.text):
            return "\0"
        return self.text[self.index+1]

    def add(self, kind: TokenType, raw: str, row: int, col: int):
        self.tokens.append(Token(kind, raw, row, col))

    def tokenize(self):
        self.tokens = []
        self.row = 1
        self.col = 1
        self.index = 0
        assert len(self.text) > 0, "Err: Invalid Input."
        while (self.index < len(self.text)):
            ch = self.peek()
            if ch in (' ', '\r', '\t'):
                self.bump()
                continue

            if ch == '\n':
                self.add(TokenType.Nl, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "=":
                row = self.row
                col = self.col
                self.bump()
                if self.peek() == "=":
                    self.add(TokenType.DbEq, ch+ch, row, col)
                    self.bump()
                else:
                    self.add(TokenType.Eq, ch, row, col)
                continue

            if ch == "-":
                self.add(TokenType.Minus, ch, self.row, self.col)
                self.bump()
                continue

            if ch == ".":
                self.add(TokenType.Dot, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "+":
                self.add(TokenType.Plus, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "(":
                self.add(TokenType.LParen, ch, self.row, self.col)
                self.bump()
                continue

            if ch == ")":
                self.add(TokenType.RParen, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "{":
                self.add(TokenType.LCurly, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "}":
                self.add(TokenType.RCurly, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "*":
                self.add(TokenType.Star, ch, self.row, self.col)
                self.bump()
                continue

            if ch == "&":
                start_row, start_col = self.row, self.col
                self.bump()
                if self.peek() == "&":
                    self.bump()
                    self.add(TokenType.DoubleAmp, "&&", start_row, start_col)
                else:
                    self.add(TokenType.Amp, "&", start_row, start_col)
                continue

            if ch == "/":
                next = self.peek_next()
                if next == "/":
                    while self.peek() not in ('\n', '\0'):
                        self.bump()
                    continue
                elif next == "*":
                    line = self.row
                    col = self.col
                    self.bump()  # /
                    self.bump()  # *
                    next = self.peek()
                    while not (next == "*" and self.peek_next() == "/") and next != "\0":
                        self.bump()
                        next = self.peek()
                    if self.peek() == "\0":
                        raise AssertionError(self.errstr(
                            line, col, "Unterminated block comment"))
                    self.bump()  # *
                    self.bump()  # /
                    continue
                else:
                    self.add(TokenType.Divide, ch, self.row, self.col)
                    self.bump()
                    continue

            if self.is_num(ch):
                num = ""
                peeked = ch
                row = self.row
                col = self.col
                while self.is_num(peeked):
                    num += self.bump()
                    peeked = self.peek()
                self.add(TokenType.Number, num, row, col)
                continue

            if self.is_alpha(ch):
                ident = ""
                peeked = ch
                row = self.row
                col = self.col
                while self.is_alphanum(peeked):
                    ident += self.bump()
                    peeked = self.peek()
                keyword = self.KEYWORDS.get(ident)
                self.add(
                    keyword if keyword is not None else TokenType.Ident, ident, row, col)
                continue

            raise AssertionError(self.errstr(
                self.row, self.col, "Unknown character"))

        self.add(TokenType.Eof, "EOF", self.row, self.col)
        return self.tokens

    def errstr(self, line: int, col: int, msg: str):
        lines = self.text.split('\n')
        final = f"At {line}:{col}\n"
        if line-3 >= 0:
            final += f" {line-2} | {lines[line-3]}\n"
        if line-2 >= 0:
            final += f" {line-1} | {lines[line-2]}\n"
        final += f" {line} | {lines[line-1]}\n"
        final += "".rjust(len(f" {line} | ") + col-1, " ")
        final += f"^ {msg}"
        return final

    def debug_print(self):
        print(self.text)
        assert len(self.tokens) > 0, "Cannot debug print with empty token array."
        largest_tok = max(self.tokens, key=lambda t: len(t.raw))
        target_len = 8
        final = "[\n"
        indent_level = 1
        for token in self.tokens:
            if token.kind == TokenType.RCurly:
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
            if token.kind == TokenType.LCurly:
                indent_level += 1
        if len(final) >= 2:
            final = final[:-2]

        final += "\n]"
        print(final)

    def _escape(self, s: str) -> str:
        return s.encode('unicode_escape').decode('ascii')

    def tokens_debug(self) -> str:
        lines = []
        for t in self.tokens:
            lines.append(
                f'Token {{ kind: {t.kind.name}, raw: "{self._escape(t.raw)}", pos: { t.row}:{t.col} }}'
            )
        return "\n".join(lines)

    def tokens_pretty_gutter(self) -> str:
        src_lines = self.text.splitlines(keepends=False)
        gutter = []
        for i, line in enumerate(src_lines, start=1):
            gutter.append(f"{i:>4} | {line}")
            carets = [" "]*len(line)
            for t in self.tokens:
                if t.row == i and t.raw and t.raw != "\n":
                    start = t.col - 1
                    end = start + max(1, len(t.raw))
                    for j in range(start, min(end, len(line))):
                        carets[j] = "^"
            if any(c != " " for c in carets):
                gutter.append("     | " + "".join(carets))
        listing = self.tokens_debug()
        return "\n".join(gutter) + "\n\n" + listing

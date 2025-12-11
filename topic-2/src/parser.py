from src.tokenizer import Token, TokenType
from dataclasses import dataclass
from typing import Protocol, Any, List
from src.types import Span
from src.source_map import SourceMap


# TODO: EBNF

class Node(Protocol):
    span: Span


@dataclass(frozen=True)
class Program(Node):
    body: List["Stmt"]
    span: Span


class Stmt(Node):
    span: Span


@dataclass(frozen=True)
class LetStmt(Stmt):
    assign: "Assign"
    span: Span


@dataclass(frozen=True)
class IfStmt(Stmt):
    condition: "Expr"
    then_block: List["Stmt"]
    else_block: List["Stmt"]


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: "Expr"
    span: Span


class Expr(Node):
    span: Span


@dataclass(frozen=True)
class PropertyAccess(Expr):
    obj: Expr
    name: Token
    span: Span


@dataclass(frozen=True)
class Literal(Expr):
    value: Any
    span: Span


@dataclass(frozen=True)
class Unary(Expr):
    op: Token
    expr: Expr
    span: Span


@dataclass(frozen=True)
class Assign(Expr):
    target: Expr
    value: Expr
    span: Span


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    op: Token
    right: Expr
    span: Span


@dataclass(frozen=True)
class Comparison(Expr):
    left: Expr
    right: Expr
    span: Span


@dataclass(frozen=True)
class Variable(Expr):
    name: Token
    span: Span


@dataclass(frozen=True)
class Grouping(Expr):
    expr: Expr
    span: Span


@dataclass(frozen=True)
class CallExpr(Expr):
    callee: Expr
    args: List[Expr]
    span: Span


class Parser:
    tokens: list[Token]
    index: int
    sm: SourceMap

    def __init__(self, tokens: list[Token], source_map: SourceMap):
        self.tokens = tokens
        self.index = 0
        self.sm = source_map

    def advance(self) -> Token:
        assert self.index < len(
            self.tokens), "tried advancing pased the last token"
        tmp = self.tokens[self.index]
        self.index += 1
        return tmp

    def peek(self):
        assert self.index < len(self.tokens), "tried peeking after last token"
        return self.tokens[self.index]

    def at(self, kind: TokenType) -> bool:
        return self.peek().kind == kind

    def expect(self, kind: TokenType, msg: str) -> Token:
        if not self.at(kind):
            t = self.peek()
            raise AssertionError(self.err_here(t, f"{msg} (got {t.kind})"))
        return self.advance()

    def consume_terminators(self):
        while self.at(TokenType.Nl):
            self.advance()

    def parse_expr(self) -> Expr:
        left = self.parse_binary(0)
        if self.at(TokenType.Eq):
            eq = self.advance()
            value = self.parse_expr()
            if not isinstance(left, (Variable, PropertyAccess)):
                raise AssertionError(
                    self.to_err(left, "invalid assignment target at {left.span}"))
            return Assign(target=left, value=value, span=Span(left.span.start, value.span.end))
        elif self.at(TokenType.DbEq):
            dbeq = self.advance()
            right = self.parse_expr()
            return Comparison(left=left, right=right, span=Span(left.span.start, right.span.end))
        elif self.at(TokenType.NEq):
            neq = self.advance()
            right = self.parse_expr()
            return Unary(op=Token(kind=TokenType.Bang, raw="!", span=neq.span), expr=Comparison(left=left, right=right, span=Span(left.span.start, right.span.end)), span=Span(left.span.start, right.span.end))
        return left

    def parse_binary(self, min_prec: int) -> Expr:
        left = self.parse_unary()
        while True:
            tok = self.peek()
            prec = self.precedence(tok)
            if prec < min_prec:
                break
            op = self.advance()
            right = self.parse_binary(prec + 1)
            left = Binary(left=left, right=right, op=op,
                          span=Span(left.span.start, right.span.end))
        return left

    _PREC = {
        TokenType.Plus: 1, TokenType.Minus: 1,
        TokenType.Star: 2, TokenType.Divide: 2,
    }

    def precedence(self, tok: Token):
        return self._PREC.get(tok.kind, -1)

    def parse_unary(self) -> Expr:
        if self.at(TokenType.Minus) or self.at(TokenType.Bang):
            op = self.advance()
            expr = self.parse_unary()
            return Unary(op=op, expr=expr, span=Span(op.span.start, expr.span.end))
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        primary = self.parse_primary()
        while True:
            if self.at(TokenType.OpenParen):
                self.advance()  # left paren
                args: List[Expr] = []
                if not self.at(TokenType.CloseParen):
                    args.append(self.parse_expr())
                    while self.at(TokenType.Comma):
                        self.advance()
                        args.append(self.parse_expr())
                r = self.expect(TokenType.CloseParen,
                                "expected )")  # right paren
                primary = CallExpr(callee=primary, args=args,
                                   span=Span(primary.span.start, r.span.end))
                continue
            if self.at(TokenType.Dot):
                self.advance()
                ident = self.expect(TokenType.Ident,  "expected ident")
                primary = PropertyAccess(obj=primary, name=ident, span=Span(
                    primary.span.start, ident.span.end))
                continue
            break
        return primary

    def parse_primary(self):
        tok = self.peek()
        match tok.kind:
            case TokenType.Number:
                tok = self.advance()
                raw = tok.raw
                val = int(raw) if raw.isdigit() else float(raw)
                return Literal(value=val, span=tok.span)
            case TokenType.Ident:
                tok = self.advance()
                return Variable(name=tok, span=tok.span)
            case TokenType.OpenParen:
                lparen = self.advance()
                inner = self.parse_expr()
                rparen = self.expect(TokenType.CloseParen, "expected )")
                return Grouping(expr=inner, span=Span(lparen.span.start, rparen.span.end))
            case TokenType.String:
                tok = self.advance()
                raw = tok.raw
                return Literal(value=raw, span=tok.span)
            case _:
                raise AssertionError(self.err_here(
                    tok, f"unexpected token {tok.kind}"))

    def parse_stmt(self) -> Stmt:
        """
        stmt := expr_stmt
        """
        token = self.peek()
        start = token.span.start
        match token.kind:
            case TokenType.Let:
                self.advance()
                assign = self.parse_expr()
                if not isinstance(assign, Assign):
                    raise AssertionError(self.to_err(
                        assign, f"incorrect lhs node: {assign}"))
                self.expect(TokenType.Semi, "expected ;")
                return LetStmt(assign=assign, span=Span(start, assign.span.end))
            case TokenType.If:
                self.advance()
                expr = self.parse_expr()
                token = self.expect(TokenType.OpenCurly, "expected {")
                self.consume_terminators()
                stmts: List[Stmt] = []
                while token.kind != TokenType.CloseCurly:
                    stmts.append(self.parse_stmt())
                    self.consume_terminators()
                    token = self.advance()
                    if token.kind == TokenType.Eof:
                        raise AssertionError(self.err_here(
                            "reached EOF before closing"))
                return IfStmt(condition=expr, then_block=stmts, else_block=[])
            case _:
                expr = self.parse_expr()
                self.expect(TokenType.Semi, "expected ;")
                return ExprStmt(expr=expr, span=Span(start, expr.span.end))

    def parse(self):
        start = self.peek().span.start
        body: List[Stmt] = []
        while not self.at(TokenType.Eof):
            self.consume_terminators()
            body.append(self.parse_stmt())
            self.consume_terminators()
        end = self.peek().span.end
        return Program(body=body, span=Span(start, end))

    def err_here(self, tok: Token, msg: str) -> str:
        (sline, scol), (eline, ecol) = self.sm.span_to_lc(tok.span)
        lines = self.sm.text.splitlines(keepends=False)

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

    def to_err(self, node: Node, msg: str) -> str:
        span = getattr(node, "span", node)

        (sline, scol), (eline, ecol) = self.sm.span_to_lc(span)
        lines = self.sm.text.splitlines(keepends=False)

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

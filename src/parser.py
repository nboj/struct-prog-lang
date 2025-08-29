from src.tokenizer import Token, TokenType
from dataclasses import dataclass
from typing import Protocol, Any, List
from src.types import Span


class Node(Protocol):
    span: Span


@dataclass(frozen=True)
class Program(Node):
    body: List["Stmt"]
    span: Span


class Stmt(Node):
    span: Span


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: "Expr"
    span: Span


class Expr(Node):
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

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

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
            raise AssertionError(f"{msg} (got {t.kind} at {t.span})")
        return self.advance()

    def consume_terminators(self):
        while self.at(TokenType.Nl):
            self.advance()

    def parse_expr(self) -> Expr:
        left = self.parse_binary(0)
        if self.at(TokenType.Eq):
            eq = self.advance()
            value = self.parse_expr()
            if not isinstance(left, (Variable,)):
                raise AssertionError(
                    f"invalid assignment target at {left.span}")
            return Assign(target=left, value=value, span=Span(left.span.start, value.span.end))
        # TODO: add eq
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
            if self.at(TokenType.LParen):
                self.advance()  # left paren
                args: List[Expr] = []
                if not self.at(TokenType.RParen):
                    args.append(self.parse_expr())
                    while self.at(TokenType.Comma):
                        self.advance()
                        args.append(self.parse_expr())
                r = self.expect(TokenType.RParen,
                                "expected )")  # right paren
                primary = CallExpr(callee=primary, args=args,
                                   span=Span(primary.span.start, r.span.end))
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
            case TokenType.LParen:
                lparen = self.advance()
                inner = self.parse_expr()
                rparen = self.expect(TokenType.RParen, "expected )")
                return Grouping(expr=inner, span=Span(lparen.span.start, rparen.span.end))
            case _:
                raise AssertionError(f"unexpected token {
                                     tok.kind} at {tok.span}")

    def parse_stmt(self) -> Stmt:
        """
        stmt := expr_stmt
        """
        token = self.peek()
        start = token.span.start
        match token.kind:
            case _:
                expr = self.parse_expr()
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

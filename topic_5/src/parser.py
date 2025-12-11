from .tokenizer import Token, TokenType
from dataclasses import dataclass
from typing import Protocol
from .types import NIL, Nil, Span
from .source_map import SourceMap


# TODO: EBNF

class Node(Protocol):
    span: Span


@dataclass(frozen=True)
class Program(Node):
    body: list["Stmt"]
    span: Span


class Stmt(Node):
    span: Span

@dataclass(frozen=True)
class BreakStmt(Stmt):
    span: Span

@dataclass(frozen=True)
class ContinueStmt(Stmt):
    span: Span

@dataclass(frozen=True)
class LetStmt(Stmt):
    assign: "Assign"
    span: Span


@dataclass(frozen=True)
class BlockStmt(Stmt):
    stmts: list[Stmt]
    span: Span


@dataclass(frozen=True)
class IfStmt(Stmt):
    condition: "Expr"
    then_block: list["Stmt"]
    else_block: list["Stmt"]

@dataclass(frozen=True)
class WhileStmt(Stmt):
    condition: "Expr"
    block: BlockStmt
    span: Span

@dataclass(frozen=True)
class FunctionStmt(Stmt):
    name: Token
    args: list["Variable"]
    block: BlockStmt
    span: Span

@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: "Expr"
    span: Span

@dataclass(frozen=True)
class ReturnStmt(Stmt):
    expr: "None | Expr"
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
    value: int | str | bool | float | Nil
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
class AugAssign(Expr):
    target: Expr
    op: Token
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
    args: list[Expr]
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
        assert self.index < len(self.tokens), "tried advancing pased the last token"
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
            raise AssertionError(self.sm.err_here(t, f"{msg} (got {t.kind})"))
        return self.advance()

    def consume_terminators(self):
        while self.at(TokenType.Nl):
            _ = self.advance()

    def parse_expr(self) -> Expr:
        left = self.parse_binary(0)
        if self.at(TokenType.Eq):
            _eq = self.advance()
            value = self.parse_expr()
            if not isinstance(left, (Variable, PropertyAccess)):
                raise AssertionError(
                    self.sm.to_err(left, "invalid assignment target at {left.span}")
                )
            if isinstance(left, Variable) and left.name.raw == "_kentid_":
                raise AssertionError(self.sm.to_err(left, "attempted explicit assignment of _kentid_"))
            return Assign(
                target=left, value=value, span=Span(left.span.start, value.span.end)
            )
        elif self.at(TokenType.PlusEq):
            op = self.advance()
            value = self.parse_expr()
            if not isinstance(left, (Variable, PropertyAccess)):
                raise AssertionError(
                    self.sm.to_err(left, "invalid assignment target at {left.span}")
                )
            return AugAssign(left, op, value, Span(left.span.start, value.span.end))

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
            left = Binary(
                left=left,
                right=right,
                op=op,
                span=Span(left.span.start, right.span.end),
            )
        return left

    _PREC = {
        TokenType.DoublePipe: 1,
        TokenType.DoubleAmp: 1,
        TokenType.NEq: 2,
        TokenType.DbEq: 2,
        TokenType.Lt: 3,
        TokenType.Gt: 3,
        TokenType.LtEq: 3,
        TokenType.GtEq: 3,
        TokenType.Plus: 4,
        TokenType.Minus: 4,
        TokenType.Mod: 5,
        TokenType.Star: 5,
        TokenType.Divide: 5,
    }

    def precedence(self, tok: Token):
        return self._PREC.get(tok.kind, -1)

    def parse_unary(self) -> Expr:
        """
        unary = ( "-" | "!" ) unary | postfix
        """
        if self.at(TokenType.Minus) or self.at(TokenType.Bang):
            op = self.advance()
            expr = self.parse_unary()
            return Unary(op=op, expr=expr, span=Span(op.span.start, expr.span.end))
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        """
        postfix = primary { call_suffix | property_suffix }
        """
        primary = self.parse_primary()
        while True:
            if self.at(TokenType.OpenParen):
                args = self.parse_args()
                primary = CallExpr(
                    callee=primary, args=args, span=Span(primary.span.start, self.peek().span.start)
                )
                continue
            if self.at(TokenType.Dot):
                _ = self.advance()
                ident = self.expect(TokenType.Ident, "expected ident")
                primary = PropertyAccess(
                    obj=primary,
                    name=ident,
                    span=Span(primary.span.start, ident.span.end),
                )
                continue
            break
        return primary

    def parse_primary(self):
        """
        primary = <int> | <number> | <bool> | <ident> | "(" expression ")"
        """
        tok = self.peek()
        match tok.kind:
            case TokenType.Number:
                tok = self.advance()
                raw = tok.raw
                val = int(raw) if raw.isdigit() else float(raw)
                return Literal(value=val, span=tok.span)
            case TokenType.Nil:
                tok = self.advance()
                return Literal(value=NIL, span=tok.span)
            case TokenType.Ident:
                tok = self.advance()
                return Variable(name=tok, span=tok.span)
            case TokenType.OpenParen:
                lparen = self.advance()
                inner = self.parse_expr()
                rparen = self.expect(TokenType.CloseParen, "expected )")
                return Grouping(
                    expr=inner, span=Span(lparen.span.start, rparen.span.end)
                )
            case TokenType.BTrue:
                tok = self.advance()
                return Literal(value=True, span=tok.span)
            case TokenType.BFalse:
                tok = self.advance()
                return Literal(value=False, span=tok.span)
            case TokenType.String:
                tok = self.advance()
                raw = tok.raw
                return Literal(value=raw, span=tok.span)
            case _:
                raise AssertionError(self.sm.err_here(tok, f"unexpected token {tok.kind}"))

    def parse_args(self) -> list[Expr]:
        args: list[Expr] = []
        _l = self.expect(TokenType.OpenParen, "expected (")
        if not self.at(TokenType.CloseParen):
            args.append(self.parse_expr())
            while self.at(TokenType.Comma):
                _ = self.advance()
                args.append(self.parse_expr())
        _r = self.expect(TokenType.CloseParen, "expected )")
        return args

    def parse_block(self) -> BlockStmt:
        start = self.expect(TokenType.OpenCurly, "expected {")
        self.consume_terminators()
        stmts: list[Stmt] = []
        while not self.at(TokenType.CloseCurly):
            stmts.append(self.parse_stmt())
            self.consume_terminators()
        end = self.expect(TokenType.CloseCurly, "expected }")
        return BlockStmt(stmts, Span(start.span.start, end.span.end))

    def parse_stmt(self) -> Stmt:
        """
        stmt := expr_stmt
        """
        token = self.peek()
        start = token.span.start
        match token.kind:
            case TokenType.OpenCurly:
                start = self.advance()
                self.consume_terminators()
                stmts: list[Stmt] = []
                while not self.at(TokenType.CloseCurly):
                    stmts.append(self.parse_stmt())
                    self.consume_terminators()
                curly = self.expect(TokenType.CloseCurly, "expected }")
                self.consume_terminators()
                return BlockStmt(stmts, Span(start.span.start, curly.span.end))
            case TokenType.Let:
                _ = self.advance()
                assign = self.parse_expr()
                if not isinstance(assign, Assign):
                    raise AssertionError(
                        self.sm.to_err(assign, f"incorrect lhs node: {assign}")
                    )
                if not isinstance(assign.target, Variable):
                    raise AssertionError(
                        self.sm.to_err(assign, "invalid lhs in assignment")
                    )
                _ = self.expect(TokenType.Semi, "expected ;")
                return LetStmt(assign=assign, span=Span(start, assign.span.end))
            case TokenType.If:
                _ = self.advance()
                expr = self.parse_expr()
                token = self.expect(TokenType.OpenCurly, "expected {")
                self.consume_terminators()
                stmts: list[Stmt] = []
                while token.kind != TokenType.CloseCurly:
                    stmts.append(self.parse_stmt())
                    self.consume_terminators()
                    token = self.peek()
                    if token.kind == TokenType.Eof:
                        raise AssertionError(
                            self.sm.err_here(token, "reached EOF before closing")
                        )
                    self.consume_terminators()
                _ = self.advance()
                else_stmts: list[Stmt] = []
                if self.at(TokenType.Else):
                    _ = self.advance()
                    _ = self.expect(TokenType.OpenCurly, "expected {")
                    self.consume_terminators()
                    token = self.peek()
                    while token.kind != TokenType.CloseCurly:
                        else_stmts.append(self.parse_stmt())
                        self.consume_terminators()
                        token = self.peek()
                        if token.kind == TokenType.Eof:
                            raise AssertionError(
                                self.sm.err_here(token, "reached EOF before closing")
                            )
                    self.consume_terminators()
                _ = self.advance()
                return IfStmt(condition=expr, then_block=stmts, else_block=else_stmts)
            case TokenType.While:
                _ = self.advance()
                condition = self.parse_expr()
                _ = self.expect(TokenType.OpenCurly, "expected {")
                stmts: list[Stmt] = []
                self.consume_terminators()
                while not self.at(TokenType.CloseCurly) and not self.at(TokenType.Eof):
                    stmts.append(self.parse_stmt())
                    self.consume_terminators()
                if self.at(TokenType.Eof):
                    raise AssertionError(self.sm.err_here(self.peek(), "expected }"))
                block = BlockStmt(stmts=stmts, span=Span(condition.span.start, self.peek().span.end))
                end = self.expect(TokenType.CloseCurly, "expected }")
                return WhileStmt(condition=condition, block=block, span=Span(condition.span.start, end.span.end))
            case TokenType.Break:
                start = self.advance()
                end = self.expect(TokenType.Semi, "expected ;")
                return BreakStmt(Span(start.span.start, end.span.end))
            case TokenType.Continue:
                start = self.advance()
                end = self.expect(TokenType.Semi, "expected ;")
                return ContinueStmt(Span(start.span.start, end.span.end))
            case TokenType.Cauman:
                start = self.advance()
                end = self.expect(TokenType.Semi, "expected ;")
                target = Variable(Token(TokenType.Ident, "_kentid_", Span(start.span.start, start.span.end)), Span(start.span.start, start.span.end))
                value = Literal("cauman@kent.edu", Span(start.span.start, start.span.end))
                return LetStmt(Assign(target, value, Span(start.span.start, end.span.end)), Span(start.span.start, end.span.end))
            case TokenType.Fn:
                fn = self.advance()
                name = self.expect(TokenType.Ident, "expected a function name")
                args = self.parse_args()
                var_args: list[Variable] = []
                for arg in args:
                    assert isinstance(arg, Variable)
                    var_args.append(arg)
                block = self.parse_block()
                print(args)
                return FunctionStmt(name, var_args, block, Span(fn.span.start, block.span.end))
            case TokenType.Return:
                ret = self.advance()
                expr = None
                if not self.at(TokenType.Semi):
                    expr = self.parse_binary(0)
                semi = self.expect(TokenType.Semi, "expected ;")
                return ReturnStmt(expr, Span(ret.span.start, semi.span.end))
            case _:
                expr = self.parse_expr()
                _ = self.expect(TokenType.Semi, "expected ;")
                return ExprStmt(expr=expr, span=Span(start, expr.span.end))

    def parse(self):
        start = self.peek().span.start
        body: list[Stmt] = []
        while not self.at(TokenType.Eof):
            self.consume_terminators()
            body.append(self.parse_stmt())
            self.consume_terminators()
        end = self.peek().span.end
        return Program(body=body, span=Span(start, end))

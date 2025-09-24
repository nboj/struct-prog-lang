from src.parser import Program, IfStmt, Expr, Stmt, LetStmt, Binary, Unary, Variable, Comparison, CallExpr, Literal, Grouping, Assign, ExprStmt
from src.tokenizer import Token, TokenType, Span
from src.source_map import SourceMap
from typing import List, Optional, Union, Protocol, Any
from dataclasses import dataclass


class Symbol:
    symbol_id: int
    name: str

    def __init__(self, symbol_id: int, name: str):
        self.symbol_id = symbol_id
        self.name = name

    def __repr__(self):
        return f"{self.name}:{self.symbol_id}"


class Scope:
    parent: Optional["Scope"]
    children: List["Scope"]
    symbols: List[Symbol]

    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.children = []
        self.symbols = []

    def to_str(self, depth=0) -> str:
        scope = self
        tmp = ""
        tmp += "─" * depth + "Scope: " + self.symbols.__repr__() + "\n"
        for child in scope.children:
            assert isinstance(child, Scope)
            tmp += "─" * depth + child.to_str(depth + 1)
        return tmp


class BoundNode(Protocol):
    span: Span


@dataclass(frozen=True)
class BoundProgram(BoundNode):
    body: List["Stmt"]
    span: Span


class BoundStmt(BoundNode):
    span: Span


@dataclass(frozen=True)
class BoundLetStmt(BoundStmt):
    assign: "BoundAssign"
    span: Span


@dataclass(frozen=True)
class BoundIfStmt(BoundStmt):
    condition: "BoundExpr"
    then_block: List["BoundStmt"]
    else_block: List["BoundStmt"]


@dataclass(frozen=True)
class BoundExprStmt(BoundStmt):
    expr: "BoundExpr"
    span: Span


class BoundExpr(BoundNode):
    span: Span


@dataclass(frozen=True)
class BoundPropertyAccess(BoundExpr):
    obj: BoundExpr
    name: Token
    span: Span


@dataclass(frozen=True)
class BoundLiteral(BoundExpr):
    value: Any
    span: Span


@dataclass(frozen=True)
class BoundUnary(BoundExpr):
    op: Token
    expr: BoundExpr
    span: Span


@dataclass(frozen=True)
class BoundAssign(BoundExpr):
    target: BoundExpr
    value: BoundExpr
    span: Span


@dataclass(frozen=True)
class BoundBinary(BoundExpr):
    left: BoundExpr
    op: Token
    right: BoundExpr
    span: Span


@dataclass(frozen=True)
class BoundComparison(BoundExpr):
    left: BoundExpr
    right: BoundExpr
    span: Span


@dataclass(frozen=True)
class BoundVariable(BoundExpr):
    name: Symbol
    span: Span


@dataclass(frozen=True)
class BoundGrouping(BoundExpr):
    expr: BoundExpr
    span: Span


@dataclass(frozen=True)
class BoundCallExpr(BoundExpr):
    callee: BoundExpr
    args: List[BoundExpr]
    span: Span


class Binder:
    program: Program
    sm: SourceMap
    scope: Scope
    var_count: int

    def __init__(self, program: Program, source_map: SourceMap):
        self.program = program
        self.sm = source_map
        self.scope = Scope()
        self.var_count = 0

    def find_symbol(self, symbol: Symbol, scope: Optional[Scope] = None):
        if scope is None:
            return None
        for scope_symbol in scope.symbols:
            if symbol.name == scope_symbol.name:
                return scope_symbol
        return self.find_symbol(symbol, scope.parent)

    def bind_expression(self, expr: Expr):
        if isinstance(expr, Binary):
            left = self.bind_expression(expr.left)
            right = self.bind_expression(expr.right)
            return BoundBinary(left, expr.op, right, expr.span)
        elif isinstance(expr, Unary):
            return BoundUnary(expr.op, self.bind_expression(expr.expr), expr.span)
        elif isinstance(expr, Variable):
            expr_id = self.var_count
            symbol = Symbol(expr_id, expr.name.raw)
            found = self.find_symbol(symbol, self.scope)
            if found is None:
                self.var_count += 1
                self.scope.symbols.append(symbol)
            else:
                symbol = found
            return BoundVariable(symbol, expr.span)
        elif isinstance(expr, Comparison):
            left = self.bind_expression(expr.left)
            right = self.bind_expression(expr.right)
            return BoundComparison(left, right, expr.span)
        elif isinstance(expr, Literal):
            return BoundLiteral(expr.value, expr.span)
        elif isinstance(expr, Grouping):
            return self.bind_expression(expr.expr)
        else:
            raise AssertionError(f"Expr not handled in VM {expr}")

    def open_scope(self):
        new_scope = Scope(self.scope)
        self.scope.children.append(new_scope)
        self.scope = new_scope

    def close_scope(self):
        assert self.scope.parent is not None, "Scope's parent was None"
        self.scope = self.scope.parent

    def bind_if(self, stmt: IfStmt):
        condition = self.bind_expression(stmt.condition)
        self.open_scope()
        then_block = []
        for s in stmt.then_block:
            then_block.append(self.bind_stmt(s))
        self.close_scope()

        self.open_scope()
        else_block = []
        for s in stmt.else_block:
            else_block.append(self.bind_stmt(s))
        self.close_scope()
        return BoundIfStmt(condition, then_block, else_block)

    def bind_let(self, stmt: LetStmt):
        assert isinstance(stmt.assign, Assign)
        target = self.bind_expression(stmt.assign.target)
        value = self.bind_expression(stmt.assign.value)
        return BoundLetStmt(BoundAssign(target, value, stmt.assign.span), stmt.span)

    def bind_expr_stmt(self, stmt: ExprStmt):
        return BoundExprStmt(stmt.expr, stmt.span)

    def bind_stmt(self, stmt: Stmt):
        if isinstance(stmt, IfStmt):
            return self.bind_if(stmt)
        elif isinstance(stmt, LetStmt):
            return self.bind_let(stmt)
        else:
            return self.bind_expr_stmt(stmt)

    def bind(self):
        stmts: List[BoundStmt] = []
        for node in self.program.body:
            stmts.append(self.bind_stmt(node))
        print(self.scope.to_str())
        return BoundProgram(stmts, self.program.span)

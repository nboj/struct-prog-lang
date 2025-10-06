from enum import Enum
from .parser import BreakStmt, ContinueStmt, Program, IfStmt, Expr, Stmt, LetStmt, Binary, Unary, Variable, CallExpr, Literal, Grouping, Assign, ExprStmt, BlockStmt, WhileStmt
from .tokenizer import Token, TokenType
from .types import Span, Symbol, SymbolType
from .source_map import SourceMap
from typing import Protocol
from dataclasses import dataclass


class ScopeKind(Enum):
    Function = "Function"
    Loop = "Loop"
    Other = "Other"

class Scope:
    parent: "Scope | None"
    children: list["Scope"]
    symbols: list[Symbol]
    kind: ScopeKind

    def __init__(self, parent: "Scope | None" = None, kind: ScopeKind = ScopeKind.Other):
        self.parent = parent
        self.children = []
        self.symbols = []
        self.kind = kind

    def to_str(self, depth:int=0) -> str:
        scope = self
        tmp: str = ""
        tmp += "─" * depth + "Scope: " + self.symbols.__repr__() + "\n"
        for child in scope.children:
            assert isinstance(child, Scope)
            tmp += "─" * depth + child.to_str(depth + 1)
        return tmp


class BoundNode(Protocol):
    span: Span


@dataclass(frozen=True)
class BoundProgram(BoundNode):
    body: list["BoundFunction"]
    span: Span


class BoundStmt(BoundNode):
    span: Span

@dataclass(frozen=True)
class BoundBreakStmt(BoundStmt):
    span: Span

@dataclass(frozen=True)
class BoundContinueStmt(BoundStmt):
    span: Span


@dataclass(frozen=True)
class BoundBlockStmt(BoundStmt):
    stmts: list[BoundStmt]
    span: Span


@dataclass(frozen=True)
class BoundLetStmt(BoundStmt):
    assign: "BoundAssign"
    span: Span


@dataclass(frozen=True)
class BoundIfStmt(BoundStmt):
    condition: "BoundExpr"
    then_block: BoundBlockStmt
    else_block: BoundBlockStmt
    span: Span


@dataclass(frozen=True)
class BoundExprStmt(BoundStmt):
    expr: "BoundExpr"
    span: Span


@dataclass(frozen=True)
class BoundWhileStmt(BoundStmt):
    condition: "BoundExpr"
    block: BoundBlockStmt
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
    value: str | int | bool | float
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
class BoundVariable(BoundExpr):
    sym: Symbol
    span: Span


@dataclass(frozen=True)
class BoundCallExpr(BoundExpr):
    callee: BoundExpr
    args: list[BoundExpr]
    span: Span

@dataclass(frozen=True)
class BoundFunction(BoundNode):
    sym: Symbol
    args: list[Symbol]
    body: BoundBlockStmt
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
        self.scope.symbols = [
            Symbol(0, "print", SymbolType.Builtin),
        ]
        self.var_count = len(self.scope.symbols)

    def lookup_here(self, name: str):
        for sym in reversed(self.scope.symbols):
            if sym.name == name:
                return sym
        return None

    def lookup(self, name: str):
        scope = self.scope
        while scope is not None:
            for sym in reversed(scope.symbols):
                if name == sym.name:
                    return sym
            scope = scope.parent
        return None

    def declare(self, name: str, type: SymbolType | None = None):
        sym = None
        if type:
            sym = Symbol(self.var_count, name, type)
        elif self.in_function():
            sym = Symbol(self.var_count, name, SymbolType.Local)
        else:
            sym = Symbol(self.var_count, name, SymbolType.Global)
        self.var_count += 1
        self.scope.symbols.append(sym)
        return sym

    def in_function(self):
        scope = self.scope
        while scope is not None:
            if scope.kind == ScopeKind.Function:
                return True
            scope = scope.parent
        return False

    def in_loop(self):
        scope = self.scope
        while scope is not None:
            if scope.kind == ScopeKind.Function:
                return False
            elif scope.kind == ScopeKind.Loop:
                return True
            scope = scope.parent
        return False

    def bind_expression(self, expr: Expr)->BoundExpr:
        if isinstance(expr, Binary):
            left = self.bind_expression(expr.left)
            right = self.bind_expression(expr.right)
            return BoundBinary(left, expr.op, right, expr.span)
        elif isinstance(expr, Unary):
            return BoundUnary(expr.op, self.bind_expression(expr.expr), expr.span)
        elif isinstance(expr, Assign):
            target = self.bind_expression(expr.target)
            value = self.bind_expression(expr.value)
            return BoundAssign(target, value, expr.span)
        elif isinstance(expr, Variable):
            symbol = self.lookup(expr.name.raw)
            if symbol is None:
                raise AssertionError(self.sm.to_err(expr, f"Tried accessing undefined variable {expr}"))
            return BoundVariable(symbol, expr.span)
        elif isinstance(expr, Literal):
            return BoundLiteral(expr.value, expr.span)
        elif isinstance(expr, Grouping):
            return self.bind_expression(expr.expr)
        elif isinstance(expr, CallExpr):
            callee = self.bind_expression(expr.callee)
            args: list[BoundExpr] = []
            for arg in expr.args:
                args.append(self.bind_expression(arg))
            return BoundCallExpr(callee, args, expr.span)

        else:
            raise AssertionError(f"Expr not handled in VM {expr}")

    def open_scope(self, scope_kind: ScopeKind = ScopeKind.Other):
        new_scope = Scope(self.scope, scope_kind)
        self.scope.children.append(new_scope)
        self.scope = new_scope

    def close_scope(self):
        assert self.scope.parent is not None, "Scope's parent was None"
        self.scope = self.scope.parent

    def bind_if(self, stmt: IfStmt):
        condition = self.bind_expression(stmt.condition)
        self.open_scope()
        then_stmts: list[BoundStmt] = []
        for s in stmt.then_block:
            bound = self.bind_stmt(s)
            then_stmts.append(bound)
        end = condition.span.end if len(then_stmts) == 0 else then_stmts[len(then_stmts)-1].span.end
        then_block = BoundBlockStmt(then_stmts, Span(condition.span.end, end))
        self.close_scope()

        self.open_scope()
        else_stmts: list[BoundStmt] = []
        for s in stmt.else_block:
            else_stmts.append(self.bind_stmt(s))
        end = then_block.span.end if len(else_stmts) == 0 else else_stmts[len(else_stmts)-1].span.end
        else_block = BoundBlockStmt(else_stmts, Span(condition.span.end, end))
        self.close_scope()
        return BoundIfStmt(condition, then_block, else_block, Span(condition.span.start, else_block.span.end))

    def bind_let(self, stmt: LetStmt):
        assert isinstance(stmt.assign, Assign) and isinstance(stmt.assign.target, Variable)
        target = BoundVariable(self.declare(
            stmt.assign.target.name.raw), stmt.assign.target.span)
        value = self.bind_expression(stmt.assign.value)
        return BoundLetStmt(BoundAssign(target, value, stmt.assign.span), stmt.span)

    def bind_expr_stmt(self, stmt: ExprStmt):
        return BoundExprStmt(self.bind_expression(stmt.expr), stmt.span)

    def bind_block(self, stmt: BlockStmt, scope_kind: ScopeKind = ScopeKind.Other):
        stmts: list[BoundStmt] = []
        self.open_scope(scope_kind)
        for s in  stmt.stmts:
            stmts.append(self.bind_stmt(s))
        self.close_scope()
        return BoundBlockStmt(stmts, stmt.span)

    def bind_stmt(self, stmt: Stmt):
        if isinstance(stmt, IfStmt):
            return self.bind_if(stmt)
        elif isinstance(stmt, LetStmt):
            return self.bind_let(stmt)
        elif isinstance(stmt, ExprStmt):
            return self.bind_expr_stmt(stmt)
        elif isinstance(stmt, BlockStmt):
            return self.bind_block(stmt)
        elif isinstance(stmt, WhileStmt):
            condition = self.bind_expression(stmt.condition)
            return BoundWhileStmt(condition=condition, block=self.bind_block(stmt.block, ScopeKind.Loop), span=stmt.span)
        elif isinstance(stmt, BreakStmt):
            if self.in_loop():
                return BoundBreakStmt(stmt.span)
            else:
                raise AssertionError(self.sm.to_err(stmt, "tried breaking outside a loop context"))
        elif isinstance(stmt, ContinueStmt):
            if self.in_loop():
                return BoundContinueStmt(stmt.span)
            else:
                raise AssertionError(self.sm.to_err(stmt, "tried continuing outside a loop context"))
        else:
            raise AssertionError(f"Unhandled stmt in binder: {stmt}")

    def bind(self):
        stmts: list[BoundStmt] = []
        for node in self.program.body:
            stmts.append(self.bind_stmt(node))
        fns: list[BoundFunction] = []
        module_init_stmts: list[BoundStmt] = []
        for stmt in stmts:
            if isinstance(stmt, BoundFunction):
                fns.append(stmt)
            else:
                module_init_stmts.append(stmt)
        start = module_init_stmts[0].span.start if len(module_init_stmts)>0 else 0
        end = module_init_stmts[len(module_init_stmts)-1].span.end if len(module_init_stmts)>0 else 0
        fns.append(BoundFunction(self.declare("module_init", SymbolType.Function), [], BoundBlockStmt(module_init_stmts, Span(start, end)), Span(start, end)))

        return BoundProgram(fns, self.program.span)

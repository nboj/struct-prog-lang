from src.parser import Program, ExprStmt, CallExpr, Expr, Binary, Literal, Grouping, IfStmt, Comparison, Unary, LetStmt, Variable, Assign
from src.tokenizer import TokenType, Token


class VM:
    program: Program
    ip: int = 0
    stack = [{}]

    def __init__(self, program: Program):
        self.program = program

    def bump(self):
        self.ip += 1

    def run(self):
        while self.ip < len(self.program.body):
            stmt = self.program.body[self.ip]
            self.run_stmt(stmt)

    def run_stmt(self, stmt):
        if isinstance(stmt, ExprStmt):
            if isinstance(stmt.expr, CallExpr):
                if isinstance(stmt.expr.callee, Variable):
                    if stmt.expr.callee.name.raw == "print":
                        print(self.eval(stmt.expr.args[0]))
                        self.bump()
                        return
                    else:
                        raise AssertionError(f"Uhnandled builtin {
                                             stmt.expr.callee.name.raw}")
                else:
                    raise AssertionError(f"Uhnandled callee type {
                                         stmt.expr.callee}")
            elif isinstance(stmt.expr, Assign):
                if not self.eval(stmt.expr.target):
                    raise AssertionError("Tried reassignment on a None lhs")
                self.run_assignment(stmt.expr)
                self.bump()
            else:
                raise AssertionError(
                    f"Uhandled expression in ExprStmt: {stmt.expr}")
        elif isinstance(stmt, IfStmt):
            condition = self.eval(stmt.condition)
            if condition:
                for then_stmt in stmt.then_block:
                    self.run_stmt(then_stmt)
                    self.ip -= 1
            elif len(stmt.else_block) > 0:
                raise AssertionError("Else not implemented in VM")
            self.bump()
        elif isinstance(stmt, LetStmt):
            assign = stmt.assign
            self.run_assignment(assign)
            self.bump()
        else:
            raise AssertionError(f"Unhandled stmt in VM {stmt}")

    def run_assignment(self, assign: Assign):
        assert isinstance(assign.target, Variable), f"target was not a token: {assign.target}"
        value = self.eval(assign.value)
        self.stack[len(self.stack)-1][assign.target.name.raw] = value

    def eval(self, expr: Expr):
        if isinstance(expr, Binary):
            left = self.eval(expr.left)
            right = self.eval(expr.right)
            if expr.op.kind == TokenType.Plus:
                return left + right
            elif expr.op.kind == TokenType.Minus:
                return left - right
            elif expr.op.kind == TokenType.Star:
                return left * right
            elif expr.op.kind == TokenType.Divide:
                return left / right
            else:
                raise AssertionError(f"Op not supported in VM {expr.op}")
        elif isinstance(expr, Unary):
            match expr.op.kind:
                case TokenType.Bang:
                    return not self.eval(expr.expr)
                case _:
                    raise AssertionError("Unary type not handled")
        elif isinstance(expr, Variable):
            index = len(self.stack)-1
            while index >= 0:
                if expr.name.raw in self.stack[index]:
                    return self.stack[index][expr.name.raw]
                index -= 1
            return None
        elif isinstance(expr, Comparison):
            left = self.eval(expr.left)
            right = self.eval(expr.right)
            return left == right
        elif isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Grouping):
            return self.eval(expr.expr)
        else:
            raise AssertionError(f"Expr not handled in VM {expr}")

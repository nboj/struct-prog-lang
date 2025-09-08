from src.parser import Program, ExprStmt, CallExpr, Expr, Binary, Literal, Grouping, IfStmt, Comparison, Unary
from src.tokenizer import TokenType


class VM:
    program: Program
    ip: int = 0

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
            if isinstance(stmt.expr, CallExpr) and stmt.expr.callee.name.raw == "print":
                print(self.eval(stmt.expr.args[0]))
            self.bump()
        elif isinstance(stmt, IfStmt):
            condition = self.eval(stmt.condition)
            if condition:
                for then_stmt in stmt.then_block:
                    self.run_stmt(then_stmt)
                    self.ip -= 1
            elif len(stmt.else_block) > 0:
                raise AssertionError("Else not implemented in VM")

            self.bump()
        else:
            raise AssertionError(f"Unhandled stmt in VM {stmt}")

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

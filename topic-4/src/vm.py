from src.parser import Program, ExprStmt, CallExpr, Expr, Binary, Literal, Grouping, IfStmt, Comparison, Unary, LetStmt, Variable, Assign
from src.tokenizer import TokenType, Token, Span


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
                        args = []
                        for arg in stmt.expr.args:
                            args.append(self.eval(arg))
                        print(*args)
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
                    raise AssertionError(
                        f"Tried reassignment on a None lhs {stmt.expr}")
                self.run_assignment(stmt.expr)
                self.bump()
            elif isinstance(stmt.expr, Variable):
                if stmt.expr.name.raw == "cauman":
                    target = Variable(name=Token(kind=TokenType.Ident, raw="_kentid_", span=Span(0, 0)), span=Span(0, 0))
                    value = Literal(value="cauman@kent.edu", span=Span(0, 0))
                    assign = Assign(target=target, value=value, span=Span(0, 0))
                    self.run_assignment(assign=assign)
                    self.bump()
                else:
                    raise AssertionError(f"Unhandled variable expression: {stmt.expr}")
            else:
                raise AssertionError(
                    f"Uhandled expression in ExprStmt: {stmt.expr}")
        elif isinstance(stmt, IfStmt):
            condition = self.eval(stmt.condition)
            if condition:
                self.stack.append({})
                for then_stmt in stmt.then_block:
                    self.run_stmt(then_stmt)
                    self.ip -= 1
                self.stack.pop()
            elif len(stmt.else_block) > 0:
                self.stack.append({})
                for then_stmt in stmt.else_block:
                    self.run_stmt(then_stmt)
                    self.ip -= 1
                self.stack.pop()
            self.bump()
        elif isinstance(stmt, LetStmt):
            assign = stmt.assign
            self.run_assignment(assign)
            self.bump()
        else:
            raise AssertionError(f"Unhandled stmt in VM {stmt}")

    def run_assignment(self, assign: Assign):
        assert isinstance(assign.target, Variable), f"target was not a token: {
            assign.target}"
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
            elif expr.op.kind == TokenType.Gt:
                return left > right
            elif expr.op.kind == TokenType.Lt:
                return left < right
            elif expr.op.kind == TokenType.DbEq:
                return left == right
            elif expr.op.kind == TokenType.NEq:
                return left != right
            elif expr.op.kind == TokenType.DoubleAmp:
                return left and right
            elif expr.op.kind == TokenType.DoublePipe:
                return left or right
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

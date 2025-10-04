from typing import Any
from .types import NIL, CodeObject
from .types import Op


def debug_stack(stack: list[Any]):
    tmp = "["
    reached_none = False
    none_start = 0
    none_end = 0
    for idx, v in enumerate(stack):
        if v is not NIL:
            if reached_none:
                none_end = idx
                reached_none = False
                if none_end - none_start == 1:
                    tmp += str(NIL) + ", "
                else:
                    tmp += f"({none_start}..{none_end}), "
            tmp += str(v) + ", "
        elif not reached_none:
            reached_none = True
            none_start = idx
    if reached_none:
        tmp += f"({none_start}..{len(stack)-1}), "
    tmp += "]"
    return tmp


def print_builtin(*args):
    print(*args)
    return NIL


class VM:
    codeobject: CodeObject
    ip: int
    fp: int
    sp: int
    stack: list[Any]
    globals: list[Any]
    heap: list[object]
    builtins: dict[int, Any]
    ops: list[int]
    a: list[int]
    b: list[int]

    running: bool

    def __init__(self, codeobject: CodeObject):
        self.ip = 0
        self.sp = 0
        self.fp = 0
        self.stack = [NIL] * 4096
        self.heap = [NIL] * 4096
        self.globals = [NIL] * codeobject.nglobals
        self.codeobject = codeobject
        self.builtins = {0: print_builtin}
        self.running = False
        self.a = [(instr.a or 0) for instr in codeobject.code]
        self.b = [(instr.b or 0) for instr in codeobject.code]
        self.ops = [instr.op for instr in codeobject.code]

        self.DISPATCH: list[Any] = [self.undefined] * len(Op)
        self.DISPATCH[Op.CALL_BUILTIN] = self.op_call_builtin
        self.DISPATCH[Op.PUSHK] = self.op_pushk
        self.DISPATCH[Op.HALT] = self.op_halt
        self.DISPATCH[Op.ADD] = self.op_add
        self.DISPATCH[Op.MUL] = self.op_mul
        self.DISPATCH[Op.DIV] = self.op_div
        self.DISPATCH[Op.SUB] = self.op_sub

        self.DISPATCH[Op.JMP] = self.op_jmp
        self.DISPATCH[Op.STOREG] = self.op_storeg
        self.DISPATCH[Op.LOADG] = self.op_loadg
        self.DISPATCH[Op.NOP] = self.op_nop
        self.DISPATCH[Op.POPN] = self.op_popn
        self.DISPATCH[Op.JZ] = self.op_jz
        self.DISPATCH[Op.NEG] = self.op_neg
        self.DISPATCH[Op.EQ] = self.op_eq
        self.DISPATCH[Op.NEQ] = self.op_neq
        self.DISPATCH[Op.LT] = self.op_lt
        self.DISPATCH[Op.GT] = self.op_gt
        self.DISPATCH[Op.STOREG_K] = self.op_storeg_k

    def op_storeg_k(self, a: int, b: int):
        self.globals[a] = b
        self.bump()

    def op_call_builtin(self, a: int, b: int):
        args: list[object] = self.stack[self.sp - b : self.sp]
        self.sp -= b
        if self.builtins[a] is None:
            raise AssertionError(f"unhandled builtin in VM {a}")
        res = self.builtins[a](*args)
        self.push_stack(res)
        self.bump()

    def op_pushk(self, a: int, _b: int):
        self.push_stack(self.codeobject.consts[a])
        self.bump()

    def op_halt(self, _a: int, _b: int):
        self.running = False

    def op_add(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left + right)
        self.bump()

    def op_mul(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left * right)
        self.bump()

    def op_div(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left / right)
        self.bump()

    def op_sub(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left - right)
        self.bump()

    def op_jmp(self, a: int, _b: int):
        self.goto(a)

    def op_storeg(self, a: int, _b: int):
        item = self.pop_stack()
        self.write_global(a, item)
        self.bump()

    def op_loadg(self, a: int, _b: int):
        item = self.globals[a]
        self.push_stack(item)
        self.bump()

    def op_nop(self, _a: int, _b: int):
        self.bump()

    def op_popn(self, a: int, _b: int):
        self.sp -= a
        self.bump()

    # NOTE: boolean logic
    def op_jz(self, a: int, _b: int):
        cond = self.pop_stack()
        # if not self.is_truthy(cond):
        if not cond:
            self.goto(a)
        else:
            self.bump()

    def op_neg(self, _a: int, _b: int):
        val = self.pop_stack()
        self.push_stack(-val)
        self.bump()

    def op_eq(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left == right)
        self.bump()

    def op_neq(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left != right)
        self.bump()

    def op_lt(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left < right)
        self.bump()

    def op_gt(self, _a: int, _b: int):
        right = self.pop_stack()
        left = self.pop_stack()
        self.push_stack(left > right)
        self.bump()

    def undefined(self, _a: int, _b: int):
        instr = self.codeobject.code[self.ip]
        raise AssertionError(f"unhandled Op code in VM {instr.op_str(instr.op)}")

    def bump(self, amount: int = 1):
        self.ip += amount

    def goto(self, new_ip: int):
        self.ip = new_ip

    def pop_stack(self):
        self.sp -= 1
        return self.stack[self.sp]

    def push_stack(self, item: Any):
        self.stack[self.sp] = item
        self.sp += 1

    def write_global(self, index: int, item: Any):
        self.globals[index] = item

    def is_truthy(self, v: Any):
        if v is NIL:
            return False
        elif isinstance(v, bool):
            return v
        elif isinstance(v, int | float):
            return v != 0
        elif isinstance(v, str) and len(v) == 0:
            return len(v) > 0
        return True

    def run(self):
        if len(self.codeobject.code) <= 0:
            return
        self.running = True
        while self.running:
            ip = self.ip
            op = self.ops[ip]
            a = self.a[ip]
            b = self.b[ip]
            self.DISPATCH[op](a, b)

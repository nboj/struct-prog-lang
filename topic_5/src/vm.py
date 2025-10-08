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
    c: list[int]

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
        self.c = [(instr.c or 0) for instr in codeobject.code]
        self.ops = [instr.op for instr in codeobject.code]

        self.DISPATCH: list[Any] = [self.undefined] * len(Op)
        self.DISPATCH[Op.CALL_BUILTIN] = self.op_call_builtin
        self.DISPATCH[Op.PUSHK] = self.op_pushk
        self.DISPATCH[Op.HALT] = self.op_halt
        self.DISPATCH[Op.ADD] = self.op_add
        self.DISPATCH[Op.MUL] = self.op_mul
        self.DISPATCH[Op.DIV] = self.op_div
        self.DISPATCH[Op.SUB] = self.op_sub
        self.DISPATCH[Op.MOD] = self.op_mod

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
        self.DISPATCH[Op.LTEQ] = self.op_lteq
        self.DISPATCH[Op.GTEQ] = self.op_gteq
        self.DISPATCH[Op.STOREG_K] = self.op_storeg_k

        self.DISPATCH[Op.CALL] = self.op_call
        self.DISPATCH[Op.LOAD_ARG] = self.op_load_arg
        self.DISPATCH[Op.RET] = self.op_ret
        self.DISPATCH[Op.STORE] = self.op_store
        self.DISPATCH[Op.LOAD] = self.op_load


    def op_load(self, a: int, _b: int, _c: int):
        item = self.stack[self.fp+3+a]
        self.stack[self.sp] = item
        self.sp += 1
        self.ip += 1

    def op_store(self, a: int, _b: int, _c: int):
        self.sp -= 1
        item = self.stack[self.sp]
        self.stack[self.fp+3+a] = item
        self.ip += 1

    def op_ret(self, a: int, _b: int, _c: int):
        self.sp -= 1
        retv = self.stack[self.sp]
        old_fp = self.fp
        self.sp = self.stack[self.fp+1]
        self.fp = self.stack[self.fp]
        self.sp -= 1
        ret_addr = self.stack[self.sp]
        for _arg in range(self.stack[old_fp+2]):
            self.sp -= 1
        self.stack[self.sp] = retv
        self.sp += 1
        self.ip = ret_addr

    def op_load_arg(self, a: int, _b: int, _c: int):
        item = self.stack[self.fp-2-a]
        self.stack[self.sp] = item
        self.sp += 1
        self.ip += 1

    def op_call(self, addr: int, nlocals: int, c: int):
        self.stack[self.sp] = self.ip + 1
        self.sp += 1
        old_fp = self.fp
        old_sp = self.sp
        self.fp = self.sp
        self.stack[self.sp] = old_fp
        self.sp += 1
        self.stack[self.sp] = old_sp
        self.sp += 1
        self.stack[self.sp] = c
        self.sp += 1
        self.sp += nlocals
        self.ip = addr

    def op_storeg_k(self, a: int, b: int, _c: int):
        self.globals[a] = b
        self.ip += 1

    def op_call_builtin(self, a: int, b: int, _c: int):
        self.sp -= b
        if self.builtins[a] is None:
            raise AssertionError(f"unhandled builtin in VM {a}")
        self.stack[self.sp] = self.builtins[a](*self.stack[self.sp:self.sp+b])
        self.sp += 1
        self.ip += 1

    def op_pushk(self, a: int, _b: int, _c: int):
        self.stack[self.sp] = self.codeobject.consts[a] 
        self.sp += 1
        self.ip += 1

    def op_halt(self, _a: int, _b: int, _c: int):
        self.running = False

    def op_add(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left + right
        self.ip += 1

    def op_mul(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left * right
        self.ip += 1

    def op_div(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left / right
        self.ip += 1

    def op_sub(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left - right
        self.ip += 1

    def op_mod(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left % right
        self.ip += 1

    def op_jmp(self, a: int, _b: int, _c: int):
        self.ip = a

    def op_storeg(self, a: int, _b: int, _c: int):
        self.sp -= 1
        item = self.stack[self.sp]
        self.globals[a] = item
        self.ip += 1

    def op_loadg(self, a: int, _b: int, _c: int):
        item = self.globals[a]
        self.stack[self.sp] = item
        self.sp += 1
        self.ip += 1

    def op_nop(self, _a: int, _b: int, _c: int):
        self.ip += 1

    def op_popn(self, a: int, _b: int, _c: int):
        self.sp -= a
        self.ip += 1

    # NOTE: boolean logic
    def op_jz(self, a: int, _b: int, _c: int):
        self.sp -= 1
        cond = self.stack[self.sp]
        # if not self.is_truthy(cond):
        if not cond:
            self.ip = a
        else:
            self.ip += 1

    def op_neg(self, _a: int, _b: int, _c: int):
        val = self.stack[self.sp-1]
        self.stack[self.sp-1] = -val
        self.ip += 1

    def op_eq(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left==right
        self.ip+=1

    def op_neq(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left!=right
        self.ip+=1

    def op_lt(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left<right
        self.ip+=1

    def op_gt(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left>right
        self.ip+=1

    def op_gteq(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left>=right
        self.ip+=1

    def op_lteq(self, _a: int, _b: int, _c: int):
        self.sp -= 1
        right = self.stack[self.sp]
        left = self.stack[self.sp-1]
        self.stack[self.sp-1] = left<=right
        self.ip+=1

    def undefined(self, _a: int, _b: int):
        instr = self.codeobject.code[self.ip]
        raise AssertionError(f"unhandled Op code in VM {instr.op_str(instr.op)}")

    def run(self):
        if len(self.codeobject.code) <= 0:
            return
        self.running = True
        DISPATCH = self.DISPATCH
        a = self.a
        b = self.b
        c = self.c
        while self.running:
            ip = self.ip
            op = self.ops[ip]
            DISPATCH[op](a[ip], b[ip], c[ip])

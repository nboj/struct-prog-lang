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

    def __init__(self, codeobject: CodeObject):
        self.ip = 0
        self.sp = 0
        self.fp = 0
        self.stack = [NIL] * 4096
        self.heap = [NIL] * 4096
        self.globals = [NIL] * codeobject.nglobals
        self.codeobject = codeobject
        self.builtins = {
            0: print_builtin
        }

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
        elif isinstance(v, int|float):
            return v != 0
        elif isinstance(v, str) and len(v) == 0:
            return len(v) > 0
        return True

    def run(self):
        if len(self.codeobject.code) <= 0:
            return
        while True:
            instr = self.codeobject.code[self.ip]
            match instr.op:
                case Op.CALL_BUILTIN:
                    assert instr.b is not None and instr.a is not None, f"call builtin had invalid params {instr}"
                    args: list[object] = [None] * instr.b
                    for i in range(instr.b):
                        args[instr.b-1-i] = self.pop_stack()
                    if self.builtins[instr.a] is None:
                        raise AssertionError(f"unhandled builtin in VM {instr.a}")
                    res = self.builtins[instr.a](*args)
                    self.push_stack(res)
                    self.bump()
                case Op.PUSHK:
                    assert instr.a is not None and instr.b is None, f"PUSHK had invalid params {instr}"
                    self.push_stack(self.codeobject.consts[instr.a])
                    self.bump()
                case Op.HALT:
                    assert instr.a is None and instr.b is None, f"HALT had invalid params {instr}"
                    break
                case Op.ADD:
                    assert instr.a is None and instr.b is None, f"ADD had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left + right)
                    self.bump()
                case Op.MUL:
                    assert instr.a is None and instr.b is None, f"MUL had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left * right)
                    self.bump()
                case Op.DIV:
                    assert instr.a is None and instr.b is None, f"DIV had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left / right)
                    self.bump()
                case Op.SUB:
                    assert instr.a is None and instr.b is None, f"SUB had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left - right)
                    self.bump()
                case Op.JMP:
                    assert instr.a is not None and instr.b is None, f"JMP had invalid params {instr}"
                    self.goto(instr.a)
                case Op.STOREG:
                    assert instr.a is not None and instr.b is None, f"STOREG had invalid params {instr}"
                    item = self.pop_stack()
                    self.write_global(instr.a, item)
                    self.bump()
                case Op.LOADG:
                    assert instr.a is not None and instr.b is None, f"LOADG had invalid params {instr}"
                    item = self.globals[instr.a]
                    self.push_stack(item)
                    self.bump()
                case Op.NOP:
                    assert instr.a is None and instr.b is None, f"NOP had invalid params {instr}"
                    self.bump()
                case Op.POPN:
                    assert instr.a is not None and instr.b is None, f"POPN had invalid params {instr}"
                    for _i in range(instr.a):
                        self.pop_stack()
                    self.bump()

                # NOTE: boolean logic
                case Op.JZ:
                    assert instr.a is not None and instr.b is None, f"JZ had invalid params {instr}"
                    cond = self.pop_stack()
                    if not self.is_truthy(cond):
                        self.goto(instr.a)
                    else:
                        self.bump()
                case Op.EQ:
                    assert instr.a is None and instr.b is None, f"EQ had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left == right)
                    self.bump()
                case Op.OR:
                    assert instr.a is None and instr.b is None, f"OR had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(self.is_truthy(left) or self.is_truthy(right))
                    self.bump()
                case Op.AND:
                    assert instr.a is None and instr.b is None, f"AND had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(self.is_truthy(left) and self.is_truthy(right))
                    self.bump()
                case Op.NEQ:
                    assert instr.a is None and instr.b is None, f"NEQ had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left != right)
                    self.bump()
                case Op.LT:
                    assert instr.a is None and instr.b is None, f"LT had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left < right)
                    self.bump()
                case Op.GT:
                    assert instr.a is None and instr.b is None, f"GT had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left > right)
                    self.bump()
                case _:
                    raise AssertionError(f"unhandled Op code in VM {instr.op_str(instr.op)}")

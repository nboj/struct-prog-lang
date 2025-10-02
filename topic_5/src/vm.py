from typing import Any
from .types import NIL, CodeObject
from .types import Op

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
        self.stack = [None] * 4096
        self.heap = [None] * 4096
        self.globals = [None] * codeobject.nglobals
        self.codeobject = codeobject
        self.builtins = {
            0: lambda *args: 
                print(*args)
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
                    if res is None:
                        res = NIL
                    self.push_stack(NIL)
                    self.bump()
                case Op.PUSHK:
                    assert instr.a is not None and instr.b is None, f"LOADK had invalid params {instr}"
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
                    self.push_stack(left or right)
                    self.bump()
                case Op.AND:
                    assert instr.a is None and instr.b is None, f"AND had invalid params {instr}"
                    right = self.pop_stack()
                    left = self.pop_stack()
                    self.push_stack(left and right)
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
                case Op.JZ:
                    assert instr.a is not None and instr.b is None, f"JZ had invalid params {instr}"
                    cond = self.pop_stack()
                    if cond == 0:
                        self.goto(instr.a)
                    else:
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
                case _:
                    raise AssertionError(f"unhandled Op code in VM {instr.op_str(instr.op)}")
        # print(self.stack, self.sp)

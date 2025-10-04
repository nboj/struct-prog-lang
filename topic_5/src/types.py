from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import override


class SymbolType(Enum):
    Local = auto()
    Global = auto()
    Builtin = auto()


class Symbol:
    sym_type: SymbolType
    symbol_id: int
    name: str

    def __init__(self, symbol_id: int, name: str, sym_type: SymbolType):
        self.symbol_id = symbol_id
        self.name = name
        self.sym_type = sym_type

    @override
    def __repr__(self):
        return f"{self.name}:{self.symbol_id}"


@dataclass(frozen=True)
class Span:
    start: int
    end: int


class Op(IntEnum):
    ADD = 0
    SUB = auto()
    DIV = auto()
    MUL = auto()
    NEG = auto()

    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()

    LOAD = auto()
    STORE = auto()
    POPN = auto()
    PUSH = auto()
    PUSHK = auto()
    STOREG_K = auto()
    JZ = auto()
    JMP = auto()
    NOP = auto()
    LOADG = auto()
    STOREG = auto()
    CALL_BUILTIN = auto()  # NOTE: the global symbol slot, num of args to pop
    HALT = auto()

class PseudoOp(IntEnum):
    LABEL = auto()
    JZ_LABEL = auto()
    JMP_LABEL = auto()

class Nil:
    @override
    def __repr__(self) -> str:
        return "nil"

NIL = Nil()

@dataclass(frozen=True)
class Instr:
    op: Op | PseudoOp
    a: int | None = None
    b: int | None = None

    # TODO: Try to find a better way than manually defining this
    def op_str(self, op: Op | PseudoOp):
        if isinstance(op, Op):
            match op:
                case Op.LOAD:
                    return "LOAD"
                case Op.STORE:
                    return "STORE"

                case Op.ADD:
                    return "ADD"
                case Op.MUL:
                    return "MUL"
                case Op.DIV:
                    return "DIV"
                case Op.SUB:
                    return "SUB"
                case Op.NEG:
                    return "NEG"

                case Op.EQ:
                    return "EQ"
                case Op.NEQ:
                    return "NEQ"
                case Op.LT:
                    return "LT"
                case Op.GT:
                    return "GT"

                case Op.STOREG_K:
                    return "STOREG_K"
                case Op.POPN:
                    return "POPN"
                case Op.PUSH:
                    return "PUSH"
                case Op.PUSHK:
                    return "PUSHK"
                case Op.JZ:
                    return "JZ"
                case Op.JMP:
                    return "JMP"
                case Op.NOP:
                    return "NOP"
                case Op.LOADG:
                    return "LOADG"
                case Op.STOREG:
                    return "STOREG"
                case Op.CALL_BUILTIN:
                    return "CALL_BUILTIN"
                case Op.HALT:
                    return "HALT"
        else:
            match op:
                case PseudoOp.LABEL:
                    return "LABEL"
                case PseudoOp.JZ_LABEL:
                    return "JZ_LABEL"
                case PseudoOp.JMP_LABEL:
                    return "JMP_LABEL"

        
    @override
    def __repr__(self) -> str:
        return f"{self.op_str(self.op)} {self.a} {self.b}"


class SlotLayout:
    map: dict[Symbol, int]
    next_slot: int

    def __init__(self):
        self.map = {}
        self.next_slot = 0

    def ensure_slot(self, sym: Symbol):
        slot = self.map.get(sym)
        if slot is None:
            self.map[sym] = self.next_slot
            slot = self.next_slot
            self.next_slot += 1
        return slot

    def get_slot(self, sym: Symbol):
        if sym not in self.map:
            raise AssertionError(f"slot not assigned for {sym}")
        return self.map[sym]


class ModuleLayout:
    mod_id: int
    name: str
    layout: SlotLayout

    def __init__(self, name: str, mod_id: int):
        self.mod_id = mod_id
        self.layout = SlotLayout()
        self.name = name

    def ensure_global(self, sym: Symbol):
        return self.layout.ensure_slot(sym)

    def get_global(self, sym: Symbol):
        return self.layout.get_slot(sym)


@dataclass(frozen=True)
class CodeObject:
    code: list[Instr]
    consts: list[object]
    nlocals: int
    nglobals: int

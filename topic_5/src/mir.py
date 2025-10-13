from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, override

from .tokenizer import Token, TokenType
from .types import NIL, Nil, Op, Symbol, SymbolType, op_str
from .binder import (
    BoundAssign,
    BoundAugAssign,
    BoundBinary,
    BoundBreakStmt,
    BoundCallExpr,
    BoundContinueStmt,
    BoundExpr,
    BoundExprStmt,
    BoundFunction,
    BoundIfStmt,
    BoundLetStmt,
    BoundLiteral,
    BoundProgram,
    BoundReturnStmt,
    BoundStmt,
    BoundUnary,
    BoundVariable,
    BoundWhileStmt,
)

type VarKey = tuple[Literal["g", "l", "t"], int]

class MStmt:
    pass


class TermStmt:
    succ: list["Block"]

    def __init__(self, succ: list["Block"]) -> None:
        self.succ = succ

class Branch(TermStmt):
    condition: "Operand"
    true_block_id: int
    false_block_id: int
    succ: list["Block"]

    def __init__(self, condition: "Operand", true_block_id: int, false_block_id: int, succ: list["Block"]) -> None:
        self.condition = condition
        self.true_block_id = true_block_id
        self.false_block_id = false_block_id
        super().__init__(succ)

    @override
    def __repr__(self) -> str:
        return f"if {self.condition} then Goto({self.true_block_id}) else Goto({self.false_block_id})"

class Goto(TermStmt):
    block_id: int
    succ: list["Block"]

    def __init__(self, block_id: int, succ: list["Block"]) -> None:
        super().__init__(succ)
        self.block_id = block_id

    @override
    def __repr__(self) -> str:
        return f"Goto({self.block_id})"

@dataclass(frozen=True)
class Halt(TermStmt):
    succ: list["Block"]

class Place:
    id: int
    def __init__(self, id: int) -> None:
        self.id = id

    @override
    def __repr__(self) -> str:
        assert False


class Local(Place):
    sym: Symbol

    def __init__(self, id: int, sym: Symbol):
        super().__init__(id)
        self.sym = sym

    @override
    def __repr__(self) -> str:
        return f"Local({self.sym.name}{self.id})"

class Global(Place):
    sym: Symbol

    def __init__(self, id: int, sym: Symbol):
        super().__init__(id)
        self.sym = sym

    @override
    def __repr__(self) -> str:
        return f"Global({self.sym.name}{self.id})"


class RValue:
    pass


class Const(RValue):
    value: int | float | str | bool | Nil

    def __init__(self, value: int | float | str | bool | Nil) -> None:
        self.value = value

    @override
    def __repr__(self) -> str:
        return f"Const({self.value})"


class Temp(Place):
    def __init__(self, id: int):
        super().__init__(id)

    @override
    def __repr__(self) -> str:
        return f"T{self.id}"


class Copy:
    origin: Place

    def __init__(self, origin: Place):
        self.origin = origin

    @override
    def __repr__(self) -> str:
        return f"Copy({self.origin})"



@dataclass(frozen=True)
class Move:
    origin: Place


type Operand = Const | Copy | Move


class BinOp(RValue):
    op: Op
    lhs: Operand
    rhs: Operand

    def __init__(self, op: Op, lhs: Operand, rhs: Operand):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    @override
    def __repr__(self) -> str:
        return f"{op_str(self.op)} {self.lhs}, {self.rhs}"

class UnaryOp(RValue):
    op: Op
    val: Operand

    def __init__(self, op: Op, val: Operand):
        self.op = op
        self.val = val

    @override
    def __repr__(self) -> str:
        return f"{op_str(self.op)} {self.val}"


class DirectFunc:
    sym: Symbol

    def __init__(self, sym: Symbol) -> None:
        self.sym = sym

    @override
    def __repr__(self) -> str:
        return f"{self.sym.name}"


@dataclass(frozen=True)
class IndirectFunc:
    operand: Operand
    func_type: SymbolType


@dataclass(frozen=True)
class Call(RValue):
    callee: DirectFunc | IndirectFunc
    args: list[Operand]


class Assign(MStmt):
    lval: Place
    rval: RValue | Operand

    def __init__(self, lval: Place, rval: RValue | Operand):
        self.lval = lval
        self.rval = rval

    @override
    def __repr__(self):
        return f"{self.lval} = {self.rval}"

class Return(TermStmt):
    val: Operand | RValue

    def __init__(self, val: Operand | RValue) -> None:
        super().__init__([])
        self.val = val

    @override
    def __repr__(self) -> str:
        return f"Return({self.val})"



class Phi(RValue):
    incoming: list[int] # (pred_block_id, value)
    def __init__(self, incoming: list[int]) -> None:
        self.incoming = incoming

    @override
    def __repr__(self) -> str:
        return f"{self.incoming}"

class PhiAssign(MStmt):
    lval: Place
    phi: Phi
    def __init__(self, lval: Place, phi: Phi):
        self.phi = phi
        self.lval = lval

    @override
    def __repr__(self):
        return f"{self.lval} = φ{self.phi}"

class Block:
    id: int
    stmts: list[MStmt]
    term: TermStmt
    preds: list["Block"]
    phis: dict[VarKey, PhiAssign]

    def __init__(self, id: int) -> None:
        self.id = id
        self.stmts = []
        self.term = Halt([])
        self.preds = []
        self.phis = {}

    @override
    def __repr__(self) -> str:
        return f"Block({self.id})"

class CFGBuilder:
    tmp_counter: int
    block_counter: int
    local_counter: int
    fn: BoundFunction
    locals: dict[int, Local]
    blocks: list[Block]
    mir_context: "MIRContext"
    loop_stack: list[tuple[Block, Block]]

    def __init__(self, fn: BoundFunction, mir_context: "MIRContext") -> None:
        self.tmp_counter = 0
        self.block_counter = 0
        self.fn = fn
        self.locals = {}
        self.local_counter = len(self.locals)
        self.blocks = []
        self.mir_context = mir_context
        self.loop_stack = []

    def ensure_local(self, sym: Symbol):
        if sym.sym_type == SymbolType.Global:
            return self.mir_context.ensure_global(sym)
        elif sym.sym_type == SymbolType.Local:
            local = self.locals.get(sym.symbol_id)
            if not local:
                local = Local(self.local_counter, sym)
                self.locals[sym.symbol_id] = local
                self.local_counter += 1
            return local
        else:
            raise AssertionError(f"Unhandled local symbol type {sym.sym_type}")

    def get_local(self, sym: Symbol):
        if sym.sym_type == SymbolType.Global:
            return self.mir_context.ensure_global(sym)
        elif sym.sym_type == SymbolType.Local:
            local = self.locals[sym.symbol_id]
            assert local is not None, "Local was none in get local"
            return local
        else:
            raise AssertionError(f"Unhandled local symbol type {sym.sym_type}")


    def lower_to_local(self, place: BoundExpr):
        if isinstance(place, BoundVariable):
            return self.ensure_local(place.sym)
        else:
            raise AssertionError("Unhandeled place type in mir")

    def new_temp(self):
        tmp = self.tmp_counter
        self.tmp_counter += 1
        return Temp(tmp)

    def emit(self, stmt: MStmt):
        self.current_block().stmts.append(stmt)

    def lower_to_place(self, expr: BoundExpr) -> Place:
        if isinstance(expr, BoundVariable):
            return self.get_local(expr.sym)
        else:
            raise AssertionError(
                f"Unhandled expr type in lower_to_place mir: {type(expr)}"
            )

    def token_to_op(self, tok: Token) -> Op:
        match tok.kind:
            case TokenType.Plus:
                return Op.ADD
            case TokenType.Minus:
                return Op.SUB
            case TokenType.Divide:
                return Op.DIV
            case TokenType.Star:
                return Op.MUL
            case TokenType.DbEq:
                return Op.EQ
            case TokenType.NEq:
                return Op.NEQ
            case TokenType.Lt:
                return Op.LT
            case TokenType.Gt:
                return Op.GT
            case TokenType.Mod:
                return Op.MOD
            case _:
                raise AssertionError(
                    "Unhandeled TokenType in binary expression mir"
                )

    def lower_logical(self, expr: BoundBinary) -> Temp:
        match expr.op.kind:
            case TokenType.DoubleAmp:
                join_block = self.new_block()
                true_block = self.new_block()
                false_block = self.new_block()
                lhs = self.lower_to_operand(expr.left)
                self.current_block().term = Branch(condition=lhs, true_block_id=true_block.id, false_block_id=false_block.id, succ=[true_block, false_block])
                self.blocks.append(true_block)
                rhs = self.lower_to_operand(expr.right)
                self.current_block().term = Goto(block_id=join_block.id, succ=[join_block])
                self.blocks.append(false_block)
                self.current_block().term = Goto(block_id=join_block.id, succ=[join_block])
                self.blocks.append(join_block)
                tmp = self.new_temp()
                phi = Phi([true_block.id, false_block.id])
                #self.emit(Assign(tmp, phi))
                self.current_block().phis[("t",tmp.id)] = PhiAssign(tmp, phi)
                return tmp
            case TokenType.DoublePipe:
                assert False
            case _ :
                raise AssertionError(f"Unhandeled expr type in mir: {type(expr)}")

    def lower_to_operand(self, expr: BoundExpr) -> Operand:
        if isinstance(expr, BoundBinary):
            match expr.op.kind:
                case TokenType.DoubleAmp:
                    return Copy(self.lower_logical(expr))
                case TokenType.DoublePipe:
                    assert False
                case _:
                    lhs = self.lower_to_operand(expr.left)
                    rhs = self.lower_to_operand(expr.right)
                    op = self.token_to_op(expr.op)
                    tmp = self.new_temp()
                    self.emit(Assign(tmp, BinOp(op, lhs, rhs)))
                    return Copy(tmp)
        elif isinstance(expr, BoundVariable):
            return Copy(self.get_local(expr.sym))
        elif isinstance(expr, BoundLiteral):
            return Const(expr.value)
        else:
            raise AssertionError(f"Unhandeled expr type in mir: {type(expr)}")

    def lower_callee(self, callee: BoundExpr) -> DirectFunc | IndirectFunc:
        if isinstance(callee, BoundVariable):
            sym = callee.sym
            if (
                sym.sym_type == SymbolType.Builtin
                or callee.sym.sym_type == SymbolType.Function
            ):
                return DirectFunc(sym)
            else:
                raise AssertionError(
                    f"Unhandled callee symbol type in lower_callee: {type(sym.sym_type)}"
                )
        else:
            raise AssertionError(
                f"Unhandled callee type in lower_to_place mir: {type(callee)}"
            )

    def lower_expression(self, expr: BoundExpr) -> RValue | Operand:
        if isinstance(expr, BoundBinary):
            match expr.op.kind:
                case TokenType.DoubleAmp:
                    return Copy(self.lower_logical(expr))
                case TokenType.DoublePipe:
                    assert False
                case _:
                    lhs = self.lower_to_operand(expr.left)
                    rhs = self.lower_to_operand(expr.right)
                    op = None
                    op = self.token_to_op(expr.op)
                    return BinOp(op, lhs, rhs)
        if isinstance(expr, BoundLiteral):
            return Const(expr.value)
        if isinstance(expr, BoundCallExpr):
            target = self.lower_callee(expr.callee)
            args: list[Operand] = []
            for arg in expr.args:
                val = self.lower_to_operand(arg)
                args.append(val)
            return Call(target, args)
        elif isinstance(expr, BoundVariable):
            place = self.ensure_local(expr.sym)
            return Copy(place)
        elif isinstance(expr, BoundAssign):
            lval = self.lower_to_local(expr.target)
            rval = self.lower_expression(expr.value)
            self.emit(Assign(lval, rval))
            return Const(NIL)
        elif isinstance(expr, BoundUnary):
            op = None
            match expr.op.kind:
                case TokenType.Minus:
                    op = Op.NEG
                case TokenType.Bang:
                    op = Op.BANG
                case _:
                    raise AssertionError(f"unhandled op kind in unary: {expr.op.kind}")
                    

            val = self.lower_to_operand(expr.expr)
            return UnaryOp(op, val)
        else:
            raise AssertionError(
                f"Unhandeled expr type in mir lower_expression {type(expr)}"
            )

    def new_block(self):
        block = Block(self.block_counter)
        self.block_counter += 1
        return block

    def current_block(self):
        return self.blocks[len(self.blocks) - 1]

    def lower_stmt(self, stmt: BoundStmt):
        if isinstance(stmt, BoundLetStmt):
            assign = stmt.assign
            place = self.lower_to_local(assign.target)
            value = self.lower_expression(assign.value)
            self.emit(Assign(place, value))
        elif isinstance(stmt, BoundReturnStmt):
            val = None
            if stmt.expr is not None:
                val = self.lower_expression(stmt.expr)
            else:
                val = Const(NIL)
            self.current_block().term = Return(val)
        elif isinstance(stmt, BoundExprStmt):
            if isinstance(stmt.expr, BoundAugAssign):
                value = None
                if stmt.expr.op.kind == TokenType.PlusEq:
                    lhs = self.lower_to_operand(stmt.expr.target)
                    rhs = self.lower_to_operand(stmt.expr.value)
                    op = Op.ADD
                    value = BinOp(op, lhs, rhs)
                else:
                    raise AssertionError(f"unhandled op kind in bound aug assign lowering: {stmt.expr.op.kind}")

                lval = self.lower_to_local(stmt.expr.target)
                self.emit(Assign(lval, value))
            elif isinstance(stmt.expr, BoundAssign):
                _ = self.lower_expression(stmt.expr)
            else: 
                rval = self.lower_expression(stmt.expr)
                lval = self.new_temp()
                self.emit(Assign(lval, rval))
        elif isinstance(stmt, BoundIfStmt):
            condition = self.lower_to_operand(stmt.condition)
            true_block = self.new_block()
            false_block = self.new_block()
            exit_block = self.new_block()
            self.current_block().term = Branch(condition=condition, true_block_id=true_block.id, false_block_id=false_block.id, succ=[true_block, false_block])
            self.blocks.append(true_block)
            for s in stmt.then_block.stmts:
                self.lower_stmt(s)

            if isinstance(self.current_block().term, Halt):
                self.current_block().term = Goto(block_id=exit_block.id, succ=[exit_block])
            self.blocks.append(false_block)
            for s in stmt.else_block.stmts:
                self.lower_stmt(s)
            if isinstance(self.current_block().term, Halt):
                self.current_block().term = Goto(block_id=exit_block.id, succ=[exit_block])
            self.blocks.append(exit_block)
        elif isinstance(stmt, BoundWhileStmt):
            head_block = self.new_block()
            body_block = self.new_block()
            exit_block = self.new_block()
            self.current_block().term = Goto(succ=[head_block], block_id=head_block.id)
            self.blocks.append(head_block)
            condition = self.lower_to_operand(stmt.condition)
            self.current_block().term = Branch(succ=[body_block, exit_block], condition=condition, true_block_id=body_block.id, false_block_id=exit_block.id)
            self.blocks.append(body_block)
            self.loop_stack.append((head_block, exit_block))
            for s in stmt.block.stmts:
                self.lower_stmt(s)
            _ = self.loop_stack.pop()
            if isinstance(self.current_block().term, Halt):
                self.current_block().term = Goto(succ=[head_block], block_id=head_block.id)
            self.blocks.append(exit_block)
        elif isinstance(stmt, BoundBreakStmt):
            assert len(self.loop_stack) > 0
            self.current_block().term = Goto(self.loop_stack[-1][1].id, succ=[self.loop_stack[-1][1]])
        elif isinstance(stmt, BoundContinueStmt):
            assert len(self.loop_stack) > 0
            self.current_block().term = Goto(self.loop_stack[-1][0].id, succ=[self.loop_stack[-1][0]])
        else:
            raise AssertionError(f"Unhandeled expr type in lowerstmt mir {type(stmt)}")

    def build(self):
        self.blocks.append(self.new_block())
        for stmt in self.fn.body.stmts:
            self.lower_stmt(stmt)
        preds: dict[int, list[Block]] = {}
        for block in self.blocks:
            for succ in block.term.succ:
                if not preds.get(succ.id):
                    preds[succ.id] = []
                preds[succ.id].append(block)
        for block in self.blocks:
            block.preds = preds.get(block.id) or []


class MIRContext:
    globals: dict[int, Global]
    global_counter: int

    def __init__(self) -> None:
        self.globals = {}
        self.global_counter = 0

    def ensure_global(self, sym: Symbol):
        if sym.sym_type == SymbolType.Global:
            local = self.globals.get(sym.symbol_id)
            if not local:
                local = Global(self.global_counter, sym)
                self.globals[sym.symbol_id] = local
                self.global_counter += 1
            return local
        assert(False), "Invalid symbol type for a global local"



class DominatorNode:
    parent: "DominatorNode | None"
    block_ref: Block
    children: list["DominatorNode"]

    def __init__(self, block_ref: Block, parent: "DominatorNode | None" = None) -> None:
        self.parent = parent
        self.children = []
        self.block_ref = block_ref

class DomTreeBuilder:
    cfg: CFGBuilder
    entry: DominatorNode | None
    map: dict[int, DominatorNode]

    def __init__(self, cfg: CFGBuilder) -> None:
        self.cfg = cfg
        self.entry = None
        self.map = {}


    def build(self):
        blocks: list[Block] = []
        if len(self.cfg.blocks) <= 0:
            return
        entry = self.cfg.blocks[0]
        seen: set[int] = set()
        def dfs(block: Block):
            if not self.map.get(block.id):
                self.map[block.id] = DominatorNode(block)
            if block.id in seen:
                return
            seen.add(block.id)
            for b in block.term.succ:
                dfs(b)
            blocks.append(block)
        dfs(entry)
        self.entry = self.map[entry.id]
        blocks.reverse()


        rpo_index = {b.id: i for i, b in enumerate(blocks)}
        entry_node = self.map[self.entry.block_ref.id]
        entry_node.parent = entry_node  # temporary sentinel


        entry_node = self.map[self.entry.block_ref.id]
        for n in self.map.values():
            n.parent = entry_node      # sentinel so nobody is None
        entry_node.parent = entry_node # entry self-parent during computation
        changed = True
        while changed:
            changed = False
            for block in blocks[1:]:  # skip entry
                preds = [pred for pred in block.preds if pred.id in self.map]
                new_parent = None

                if len(preds) == 0:
                    continue  # only entry should have 0 preds here
                elif len(preds) == 1:
                    new_parent = self.map[preds[0].id]
                else:
                    # fast-path when all preds are same node
                    first = self.map[preds[0].id]
                    if all(self.map[p.id] is first for p in preds[1:]):
                        new_parent = first
                    else:
                        # wait until all pred parents exist
                        if any(self.map[p.id].parent is None for p in preds):
                            continue
                        cands = [self.map[p.id] for p in preds]

                        def lca(a: DominatorNode, b: DominatorNode) -> DominatorNode:
                            x, y = a, b
                            while x is not y:
                                while rpo_index[x.block_ref.id] > rpo_index[y.block_ref.id]:
                                    assert x.parent is not None
                                    x = x.parent
                                while rpo_index[y.block_ref.id] > rpo_index[x.block_ref.id]:
                                    assert y.parent is not None
                                    y = y.parent
                                if x is not y:
                                    assert x.parent is not None
                                    assert y.parent is not None
                                    x = x.parent
                                    y = y.parent
                            return x

                        acc = cands[0]
                        for d in cands[1:]:
                            acc = lca(acc, d)
                        new_parent = acc

                node = self.map[block.id]
                if new_parent is not None and node.parent is not new_parent:
                    node.parent = new_parent
                    changed = True

        self.map[self.entry.block_ref.id].parent = None
        self.entry = self.map[self.entry.block_ref.id]

        for n in self.map.values():
            n.children.clear()
        for n in self.map.values():
            if n.parent is not None:
                n.parent.children.append(n)


    def print_node(self,node: DominatorNode):
        assert node.block_ref is not None
        print()
        print("BLOCKID:", node.block_ref.id)
        if node.parent is not None:
            assert node.parent.block_ref is not None
            print("DOMINATOR:", node.parent.block_ref.id)
        tmp = "CHILDREN: ["
        for child in node.children:
            assert child.block_ref is not None
            tmp += str(child.block_ref.id) + ", "
        if len(node.children) > 0:
            tmp = tmp[0:-2]
        tmp += "]"
        print(tmp)
        print()

    def print_tree(self, node: DominatorNode):
        self.print_node(node)
        for child in node.children:
            self.print_tree(child)


class SSANamer:
    df: dict[int, set[Block]]
    cfg: CFGBuilder
    vars: dict[VarKey, list[int]]
    temp_counter: int
    def_blocks: dict[VarKey, set[Block]]
    var_info: dict[VarKey, Place]
    tree: DomTreeBuilder
    var_counter: int

    def __init__(self, df: dict[int, set[Block]], cfg: CFGBuilder, tree: DomTreeBuilder) -> None:
        self.df = df
        self.cfg = cfg
        self.vars = defaultdict(list)
        self.def_blocks = defaultdict(set)
        self.var_counter = 0
        self.temp_counter = 0
        self.var_info = {}
        self.tree = tree

    def rename_operand(self, operand: Operand):
        match operand:
            case Const():
                return operand
            case Copy(origin=origin):
                return Copy(self.rename_place(origin))
            case Move(origin=origin):
                raise AssertionError("move ops are not implemented yet")

    def rename_rval(self, rval: RValue | Operand) -> RValue | Operand:
        match rval:
            case Const():
                return self.rename_operand(rval)
            case Copy():
                return self.rename_operand(rval)
            case Move():
                return self.rename_operand(rval)
            case BinOp(op=op, lhs=lhs, rhs=rhs):
                return BinOp(op=op, lhs=self.rename_operand(lhs), rhs=self.rename_operand(rhs))
            case UnaryOp(op=op, val=val):
                return UnaryOp(op=op, val=self.rename_operand(val))
            case Call(callee, args):
                new_args: list[Operand] = []
                for arg in args:
                    new_args.append(self.rename_operand(arg))
                new_callee = None
                match callee:
                    case DirectFunc():
                        new_callee = callee
                    case IndirectFunc(operand=operand, func_type=func_type):
                        new_callee = IndirectFunc(self.rename_operand(operand), func_type=func_type)
                return Call(new_callee, new_args)
            case _:
                raise AssertionError(f"unhandled rval type: {type(rval)}")

    def topvar(self, id: VarKey):
        if len(self.vars[id]) <= 0 and id[0] == "g":
            self.vars[id].append(0)
        assert len(self.vars[id]) > 0
        return self.vars[id][-1]

    def rename_place(self, place: Place) -> Place:
        key = self.var_key_from_place(place)
        match place:
            case Temp(id=id):
                return Temp(self.topvar(key))
            case Global(id=id, sym=sym):
                return Global(self.topvar(key), sym)
            case Local(id=id, sym=sym):
                return Local(self.topvar(key), sym)
            case _:
                raise AssertionError(f"unhandled place type in renamer: {type(place)}")

    def define_place(self, place: Place) -> Place:
        key = self.var_key_from_place(place)
        match place:
            case Temp(id=id):
                self.vars[key].append(self.temp_counter)
                self.temp_counter += 1
                return Temp(self.vars[key][len(self.vars[key])-1])
            case Global(id=id, sym=sym):
                self.vars[key].append(self.var_counter)
                self.var_counter += 1
                return Global(self.vars[key][-1], sym)
            case Local(id=id, sym=sym):
                self.vars[key].append(self.var_counter)
                self.var_counter += 1
                return Local(self.vars[key][-1], sym)
            case _:
                raise AssertionError(f"unhandled place type in renamer: {type(place)}")

    def rename_stmt(self, stmt: MStmt):
        match stmt:
            case Assign(lval=lval, rval=rval):
                new_rval = self.rename_rval(rval)
                new_lval = self.define_place(lval)
                return Assign(new_lval, new_rval)
            case _:
                raise AssertionError(f"unhandled stmt kind in ssa renaming: {type(stmt)}")


    def var_key_from_place(self, place: Place) -> VarKey:
        match place:
            case Temp(id=id):
                return ("t", id)
            case Global(id=_, sym=sym):
                return ("g", sym.symbol_id)
            case Local(id=_, sym=sym):
                return ("l", sym.symbol_id)
            case _:
                raise AssertionError(f"unhandled place: {type(place)}")


    def nearest_def_block_id(self, start_block:Block, var_key: int, defines_map: dict[int, set[int]]) -> int | None:
        """Walk backward from start_block through preds until we hit a block that defines var_key.
           Return that block id, or None if no definition is found."""
        stack = [start_block]
        visited: set[int] = set()
        while stack:
            b = stack.pop()
            if b.id in defines_map[var_key]:
                return b.id
            if b.id in visited:
                continue
            visited.add(b.id)
            for p in b.preds:
                stack.append(p)
        return None

    def rename_term(self, term: TermStmt) -> TermStmt:
        match term:
            case Return(val=val):
                return Return(self.rename_rval(val))
            case Goto():
                return term
            case Branch(condition=condition, true_block_id=true_block_id, false_block_id=false_block_id, succ=succ):
                return Branch(self.rename_operand(condition), true_block_id=true_block_id, false_block_id=false_block_id, succ=succ)
            case Halt():
                return term
            case _:
                raise AssertionError(f"unhandled term type when renaming: {type(term)}")

    def rename_phi(self, phi: PhiAssign) -> PhiAssign:
        lval = self.define_place(phi.lval)
        return PhiAssign(lval, phi.phi)

    def run(self):
        print("<<<<<<<<RENAMING>>>>>>>>")
        map: dict[int, int] = {}
        blocks: list[Block] = []
        block_defines_var: dict[VarKey, set[int]] = defaultdict(set)
        for idx, block in enumerate(self.cfg.blocks):
            for stmt in block.stmts:
                match stmt:
                    case Assign(lval=lval, rval=rval):
                        k = self.var_key_from_place(lval)
                        self.var_info[k] = lval
                        self.def_blocks[k].add(block)
                        block_defines_var[k].add(block.id)
                    case _:
                        pass
            map[block.id] = idx
            blocks.append(block)
        self.def_blocks = defaultdict(set, {k: v for (k, v) in self.def_blocks.items() if len({b.id for b in v}) >= 2})
        block_defines_var = defaultdict(set, {k: v for (k, v) in block_defines_var.items() if len(v) >= 2})
        hasphi: dict[VarKey, set[int]] = defaultdict(set)                 # var -> {join_id}
        phi_sources: dict[VarKey, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

        for var, defsites in self.def_blocks.items():
            for d in defsites:
                source = d.id
                stack = [d] 
                visited: set[int] = set()

                while stack:
                    b = stack.pop()
                    for y in self.df.get(b.id, ()):
                        j_id = y.id
                        j_idx = map[j_id]

                        if j_id not in hasphi[var]:
                            proto = self.var_info[var]
                            match proto:
                                case Global(id=_, sym=sym):
                                    dest = Global(0, sym)
                                case Local(id=_, sym=sym):
                                    dest = Local(0, sym)
                                case Temp(id=id):
                                    dest = Temp(id)
                                case _:
                                    raise AssertionError("unexpected place proto")

                            blocks[j_idx].phis[var] = PhiAssign(dest, Phi([]))
                            hasphi[var].add(j_id)
                        if j_id not in visited:
                            phi_sources[var][j_id].add(source)

                        if j_id not in block_defines_var[var] and j_id not in visited:
                            visited.add(j_id)
                            stack.append(blocks[j_idx])

        for block in blocks:
            for succ in block.term.succ:
                for phi in succ.phis.values():
                    phi.phi.incoming.append(block.id)


        entry = self.tree.entry
        assert entry is not None
        def rename(entry: DominatorNode):
            new_stmts: list[MStmt] = []
            new_phis: dict[VarKey, PhiAssign] = {}
            block = entry.block_ref
            for phi in block.phis.values():
                new_phis[self.var_key_from_place(phi.lval)] = self.rename_phi(phi)

            for stmt in block.stmts:
                new_stmts.append(self.rename_stmt(stmt))
            block.phis = new_phis
            block.stmts = new_stmts
            block.term = self.rename_term(block.term)
            vars_copy = deepcopy(self.vars)
            for child in entry.children:
                rename(child)
                self.vars = deepcopy(vars_copy)
            blocks[map[block.id]] = block
        rename(entry)
        #for block in blocks:
        #    new_stmts: list[MStmt] = []
        #    new_phis: dict[VarKey, PhiAssign] = {}
        #    for phi in block.phis.values():
        #        new_phis[self.var_key_from_place(phi.lval)] = self.rename_phi(phi)

        #    for stmt in block.stmts:
        #        new_stmts.append(self.rename_stmt(stmt))
        #    block.phis = new_phis
        #    block.stmts = new_stmts
        #    block.term = self.rename_term(block.term)

        for block in blocks:
            print()
            print(block)
            for (_id, phi) in block.phis.items():
                print(phi)
            for stmt in block.stmts:
                print(stmt)
            print(block.term)
        print("===PHIS===")
        for block in blocks:
            for (id, assign) in block.phis.items():
                print(assign)
        print("===END PHIS===")
        print()
        print()
        print("===== REMOVING PHIS =====")



        def global_rename_operand_use(old_lval: Global | Local, new_lval: Global | Local, operand: Operand) -> Operand:
            match operand:
                case Const():
                    return operand
                case Copy(origin=origin):
                    match origin:
                        case Global(id=id, sym=sym):
                            if id == old_lval.id and sym == old_lval.sym:
                                return Copy(new_lval)
                            else:
                                return operand
                        case Local(id=id, sym=sym):
                            if id == old_lval.id and sym == old_lval.sym:
                                return Copy(new_lval)
                            else:
                                return operand
                        case Temp(id=id):
                            return operand
                        case _:
                            return operand

                case Move(origin=origin):
                    raise AssertionError("move ops are not implemented yet")

        def global_rename_use(old_lval: Global | Local, new_lval: Global | Local, rval: RValue | Operand) -> RValue | Operand:
            match rval:
                case Const():
                     return global_rename_operand_use(old_lval, new_lval, rval)
                case Copy():
                     return global_rename_operand_use(old_lval, new_lval, rval)
                case Move():
                     return global_rename_operand_use(old_lval, new_lval, rval)
                case BinOp(op=op, lhs=lhs, rhs=rhs):
                    return BinOp(op=op, lhs=global_rename_operand_use(old_lval, new_lval, lhs), rhs=global_rename_operand_use(old_lval, new_lval, rhs))
                case UnaryOp(op=op, val=val):
                    return UnaryOp(op=op, val=global_rename_operand_use(old_lval, new_lval, val))
                case Call(callee, args):
                    new_args: list[Operand] = []
                    for arg in args:
                        new_args.append(global_rename_operand_use(old_lval, new_lval, arg))
                    new_callee = None
                    match callee:
                        case DirectFunc():
                            new_callee = callee
                        case IndirectFunc(operand=operand, func_type=func_type):
                            new_callee = IndirectFunc(global_rename_operand_use(old_lval, new_lval, operand), func_type=func_type)
                    return Call(new_callee, new_args)
                case _:
                    raise AssertionError(f"unhandled rval type: {type(rval)}")
            


        def global_rename_stmt(old_lval: Global | Local, new_lval: Global | Local, stmt: MStmt) -> MStmt:
            match stmt:
                case Assign(lval=lval, rval=rval):
                    new_r = global_rename_use(old_lval, new_lval, rval)
                    if old_lval == lval:
                        return Assign(new_lval, new_r)
                    else:
                        return Assign(lval, new_r)

                case _:
                    raise AssertionError(f"unhandled stmt kind in ssa renaming: {type(stmt)}")

        def global_rename_term(old_lval: Global | Local, new_lval: Global | Local, term: TermStmt) -> TermStmt:
            match term:
                case Return(val=val):
                    return Return(global_rename_use(old_lval, new_lval, val))
                case Goto():
                    return term
                case Branch(condition=condition, true_block_id=true_block_id, false_block_id=false_block_id, succ=succ):
                    return Branch(global_rename_operand_use(old_lval, new_lval, condition), true_block_id=true_block_id, false_block_id=false_block_id, succ=succ)
                case Halt():
                    return term
                case _:
                    raise AssertionError(f"unhandled term type when renaming: {type(term)}")


        for block in blocks:
            if len(block.phis) > 0:
                for (id, assign) in block.phis.items():
                    phi_lval = assign.lval
                    assert isinstance(phi_lval, (Global, Local))
                    incoming = assign.phi.incoming
                    visited: set[int] = set()
                    while len(incoming) > 0:
                        new_incomings = []
                        for b in incoming:
                            if b in visited:
                                continue
                            visited.add(b)
                            found = False
                            for idx, stmt in enumerate(reversed(blocks[map[b]].stmts)):
                                match stmt:
                                    case Assign(lval=lval, rval=rval):
                                        match lval:
                                            case Local(id=id,sym=sym):
                                                pass
                                            case Global(id=id,sym=sym):
                                                pass
                                            case _:
                                                continue
                                        if lval.sym == phi_lval.sym:
                                            found = True
                                            fwd_idx = len(blocks[map[b]].stmts) - 1 - idx
                                            print(f"FOUND: {lval.sym.name}{lval.id}, REPLACE WITH:  {phi_lval.sym.name}{phi_lval.id}")
                                            print(f"STMT: {blocks[map[b]].stmts[fwd_idx]}")
                                            print(f"Starting with block: {map[b]+1}")
                                            for _block in blocks:
                                                _block.stmts = [global_rename_stmt(lval, phi_lval, s) for s in _block.stmts]
                                                _block.term  = global_rename_term(lval, phi_lval, _block.term)
                                            blocks[map[b]].stmts[fwd_idx] = Assign(phi_lval, global_rename_use(lval, phi_lval, rval))
                                            break
                                    case _:
                                        pass
                                pass
                            if not found:
                                new_incomings += [p.id for p in blocks[map[b]].preds]
                        incoming = new_incomings
                block.phis = {}

        for block in blocks:
            print()
            print(block)
            for (_id, phi) in block.phis.items():
                print(phi)
            for stmt in block.stmts:
                print(stmt)
            print(block.term)
        print("<<<<<<<<RENAMING FINISHED>>>>>>>>")
        return blocks


class Mir:
    program: BoundProgram
    temp_counter: int
    debug: bool
    cfgs: list[CFGBuilder]
    dom_trees: list[DomTreeBuilder]
    context: MIRContext
    DFs: list[dict[int, set[Block]]] = []

    def __init__(self, bound_program: BoundProgram, debug: bool = False) -> None:
        self.program = bound_program
        self.temp_counter = 0
        self.debug = debug
        self.cfgs = []
        self.dom_trees = []
        self.context =  MIRContext()

    def build_cfg(self):
        for fn in self.program.body:
            builder = CFGBuilder(fn, self.context)
            builder.build()
            self.cfgs.append(builder)

    def build_trees(self):
        for cfg in self.cfgs:
            tree = DomTreeBuilder(cfg)
            tree.build()
            self.dom_trees.append(tree)

    def compute_dominator_frontiers(self):
        for tree in self.dom_trees:
            df_map: dict[int, set[Block]] = {}
            for block in tree.cfg.blocks:
                dominator = tree.map.get(block.id)
                if not dominator:
                    raise AssertionError("unreachable code found when computing dominance frontiers")
                if dominator.parent is None:
                    continue
                parent = dominator.parent.block_ref
                current_preds: list[Block] = block.preds
                if len(block.preds) < 2:
                    continue
                while True:
                    next_preds: list[Block] = []
                    for pred in current_preds:
                        if parent != pred:
                            if df_map.get(pred.id) is None:
                                df_map[pred.id] = set()
                            df_map[pred.id].add(block)
                            _parent = tree.map[pred.id].parent
                            assert _parent is not None
                            assert isinstance(_parent.block_ref, Block)
                            next_preds.append(_parent.block_ref)
                    if len(next_preds) == 0:
                        break
                    current_preds = next_preds
            self.DFs.append(df_map)

    def rename(self):
        for (cfg, df, tree) in zip(self.cfgs, self.DFs, self.dom_trees):
            namer = SSANamer(df=df, cfg=cfg, tree=tree)
            blocks = namer.run()

    def lower(self):
        self.build_cfg()
        self.build_trees()
        self.compute_dominator_frontiers()
        self.rename()


if __name__ == "__main__":
    from .tokenizer import Tokenizer
    from .parser import Parser
    from .binder import Binder

    text = """
let a = 0;
let a = 2;
print("test", a);
//// ==============================
//// Big SSA / φ / Liveness Testbed
//// ==============================
//
//// ---- Globals used across tests ----
//let p = true;
//let q = true;
//let r = true;
//let s = true;
//let t = true;
//
//let a = 0;
//let b = 0;
//let c = 0;
//let d = 0;
//let e = 0;
//
//let x = 0;
//let y = 0;
//let z = 0;
//let w = 0;
//let v = 0;
//let u = 0;
//
//let n = 0;
//let m = 0;
//
//let flag = false;
//
//// ---------- Helper fns (zero-arg) ----------
//fn set_small_constants() {
//    a = 1;
//    b = 2;
//    c = 3;
//    d = 4;
//    e = 5;
//}
//
//fn set_flags_pttff() {
//    p = true;  q = true;  r = true;  s = false; t = false;
//}
//
//fn flip_flags() {
//    p = !p; q = !q; r = !r; s = !s; t = !t;
//}
//
//fn bump_xy() {
//    x = x + 1;
//    y = y + 2;
//}
//
//fn write_v_from_w() {
//    v = w + 10;
//}
//
//fn id_x_into_z() {
//    // tests copy-through with no local reassign in some paths
//    z = x;
//}
//
//fn print_avxwy() {
//    print("a,b,x,w,y:", a, b, x, w, y);
//}
//
//// ============ Test 1 ============
//// Simple diamond, both arms assign -> φ at join
//fn test_phi_both_arms() {
//    a = 0;
//    if true {
//        a = 1;
//    } else {
//        a = 2;
//    }
//    print("T1 a:", a); // expect 1
//}
//
//// ============ Test 2 ============
//// Diamond where ELSE arm does not write -> φ(a_from_then, a_from_entry)
//fn test_phi_one_arm() {
//    a = 10;
//    if p {           // p starts true in module_init
//        a = 11;
//        print("T2 side:", a);
//    } else {
//        // no write to a here
//    }
//    print("T2 a:", a); // expect 11
//}
//
//// ============ Test 3 ============
//// Nested diamonds that feed another φ
//fn test_nested_phi_chain() {
//    set_small_constants(); // a=1,b=2,c=3,d=4,e=5
//    // x comes from (a or b)
//    if p {
//        x = a;
//    } else {
//        x = b;
//    }
//    // y comes from (b or x)
//    if q {
//        y = b;
//    } else {
//        y = x;
//    }
//    // z comes from (x or y)
//    if r {
//        z = x;
//    } else {
//        z = y;
//    }
//    print("T3 x,y,z:", x, y, z); // expect x=1, y=2, z=1 (with p=true,q=true,r=true)
//}
//
//// ============ Test 4 ============
//// Loop-carried value with back-edge φ (simple increment loop)
//fn test_loop_simple_induction() {
//    a = 0;
//    while a < 5 {
//        a = a + 1;
//    }
//    print("T4 a:", a); // expect 5
//}
//
//// ============ Test 5 ============
//// Loop with inner diamond on the back-edge (even/odd bump), classic from the thread
//fn test_loop_nested_diamond_backedge() {
//    a = 0;
//    while a < 6 {
//        if a % 2 == 0 {
//            a = a + 2;
//        } else {
//            a = a + 1;
//        }
//    }
//    print("T5 a:", a); // expect 6
//}
//
//// ============ Test 6 ============
//// Loop where only one arm writes; the other carries the previous value
//fn test_loop_one_arm_writes() {
//    x = 0;
//    n = 4;
//    while x < n {
//        if p {
//            // then arm writes
//            x = x + 1;
//        } else {
//            // else arm does NOT write x (edge-live φ)
//            // (keep this empty intentionally)
//        }
//        // flip p to force both edges to be taken across iterations
//        p = !p;
//    }
//    print("T6 x:", x); // expect 4
//}
//
//// ============ Test 7 ============
//// Two different φs joining simultaneously, then used together
//fn test_two_phis_together() {
//    set_small_constants(); // a=1,b=2,c=3,d=4,e=5
//
//    // φ #1 -> x
//    if p { x = a; } else { x = b; }
//
//    // φ #2 -> y
//    if q { y = d; } else { y = e; }
//
//    w = x + y;
//    print("T7 x,y,w:", x, y, w); // with p,q=true: x=1,y=4,w=5
//}
//
//// ============ Test 8 ============
//// Multiple writes in same pred before the join (ensure "last-def" is chosen)
//fn test_last_def_in_pred() {
//    a = 0;
//    if p {
//        a = 4;
//        a = a + 2;     // last def in THEN
//    } else {
//        a = a + 1;     // last def in ELSE (uses prior a)
//    }
//    print("T8 a:", a); // with p=true: expect 6
//}
//
//// ============ Test 9 ============
//// Copy-through across join with no intervening write in one path
//fn test_copy_through_join() {
//    set_small_constants(); // a=1, b=2
//    if p {
//        x = a;
//    } else {
//        // no write to x here
//    }
//    print("T9 x:", x); // expect 1
//}
//
//// ============ Test 10 ============
//// Interprocedural: function writes globals that feed into φs later
//fn test_interprocedural_globals() {
//    set_small_constants(); // a..e
//    bump_xy();             // x=1,y=2
//    id_x_into_z();         // z=x
//    if q {
//        w = z;         // write in THEN
//    } else {
//        // no write in ELSE
//    }
//    write_v_from_w();      // v = w + 10 (tests read after φ-like merge)
//    print("T10 x,y,z,w,v:", x, y, z, w, v); // with q=true: x=1,y=2,z=1,w=1,v=11
//}
//
//// ============ Test 11 ============
//// Diamond feeding diamond feeding loop-carried φ
//fn test_diamond_chain_into_loop() {
//    set_small_constants(); // a=1,b=2,c=3,d=4,e=5
//    // First diamond → x
//    if p { x = a; } else { x = b; }
//    // Second diamond → y (mix x and c)
//    if r { y = x; } else { y = c; }
//
//    // Loop carries z from (y or z+1)
//    z = 0;
//    n = 3;
//    while z < n {
//        if s {
//            z = y;        // write z from merged y (tests φ with incoming from preheader)
//        } else {
//            z = z + 1;    // normal increment
//        }
//        s = !s; // alternate
//    }
//    print("T11 x,y,z:", x, y, z); // with p,r=true, s flips: y=1, z likely 1 then increments; final >=3
//}
//
//// ============ Test 12 ============
//// Two loop-carried variables updated in different arms
//fn test_two_carriers() {
//    x = 0; y = 0; n = 5;
//    while x < n {
//        if x % 2 == 0 {
//            x = x + 1;   // writes x
//            // y not written
//        } else {
//            y = y + 3;   // writes y
//            x = x + 1;
//        }
//    }
//    print("T12 x,y:", x, y); // expect x=5, y= (3*#odd-iters) = 3*2 = 6
//}
//
//
//// ============ Test 14 ============
//// Temp values feeding φs, then used after φ removal
//fn test_temps_and_phis() {
//    set_small_constants(); // a=1,b=2,c=3
//    if p {
//        // create a temp-like chain: t = a+b; x = t + 1;
//        x = (a + b) + 1;
//    } else {
//        // other arm: x = c + 2;
//        x = c + 2;
//    }
//    // join
//    print("T14 x:", x); // with a=1,b=2 -> x=4
//}
//
//// ============ Test 15 ============
//// Deep nesting mix (stress)
//fn test_deep_mix() {
//    set_flags_pttff(); // p=true,q=true,r=true,s=false,t=false
//    set_small_constants(); // a..e = 1..5
//    u = 0; v = 0; w = 0; x = 0; y = 0; z = 0;
//
//    if p {
//        if q {
//            x = a;
//            if r {
//                y = b;
//            } else {
//                y = c;
//            }
//        } else {
//            x = d;
//            y = e;
//        }
//    } else {
//        if s {
//            x = b + c;
//        } else {
//            // no write to x
//        }
//        if t {
//            y = a + d;
//        } else {
//            // no write to y
//        }
//    }
//
//    // Another layer of diamonds that touch x,y
//    if r {
//        z = x + y;
//    } else {
//        z = x + 1;
//    }
//
//    // A tiny loop that toggles s, adds to w from z or y
//    m = 3;
//    while m > 0 {
//        if s {
//            w = z + y;
//        } else {
//            w = w + 1;
//        }
//        s = !s;
//        m = m - 1;
//    }
//
//    print("T15 x,y,z,w:", x, y, z, w);
//}
//
//// ============ Driver (module_init) ============
//print("=== RUN TESTS START ===");
//
//test_phi_both_arms();
//test_phi_one_arm();
//test_nested_phi_chain();
//test_loop_simple_induction();
//test_loop_nested_diamond_backedge();
//test_loop_one_arm_writes();
//test_two_phis_together();
//test_last_def_in_pred();
//test_copy_through_join();
//test_interprocedural_globals();
//test_diamond_chain_into_loop();
//test_two_carriers();
//test_temps_and_phis();
//test_deep_mix();
//
//print("=== RUN TESTS END ===");
"""
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens, tokenizer.sm)
    parsed = parser.parse()
    binder = Binder(parsed, tokenizer.sm)
    bound = binder.bind()
    mir = Mir(bound, debug=True)
    lowered = mir.lower()
    for cfg in mir.cfgs:
        print(f"NEW CFG: {cfg.fn.sym.name}")
        for block in cfg.blocks:
            print(f"BLOCK: {block.id}")
            tmp = "PREDS: ["
            for pred in block.preds:
                tmp += str(pred.id) + ", "
            if len(block.preds) > 0:
                tmp = tmp[0:-2]
            tmp += "]"
            print(tmp)
            print("STMTS:")
            for stmt in block.stmts:
                print(stmt)
            print("TERM:")
            print(block.term)
            print()
        print()

    print()
    print()
    print()
    for idx, (tree, cfg) in enumerate(zip(mir.dom_trees, mir.cfgs)):
        print(f"NEW FUNCTION: {cfg.fn.sym.name}")
        assert tree.entry is not None
        tree.print_tree(tree.entry)

        print("=================")
        df_map = mir.DFs[idx]
        print("JOIN:", len(df_map))
        for (id, items) in df_map.items():
            tmp = f"DF={id}, ["
            for item in items:
                tmp += str(item.id) + ", "
            if len(items) > 0:
                tmp = tmp[0:-2]
            tmp += "]"
            print(tmp)
        print("=================")
        print()

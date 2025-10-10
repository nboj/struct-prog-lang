from collections import defaultdict
from dataclasses import dataclass
from hmac import new
from typing import override

from .tokenizer import Token, TokenType
from .types import NIL, Nil, Op, Symbol, SymbolType, op_str
from .binder import (
    BoundAssign,
    BoundAugAssign,
    BoundBinary,
    BoundCallExpr,
    BoundExpr,
    BoundExprStmt,
    BoundFunction,
    BoundIfStmt,
    BoundLetStmt,
    BoundLiteral,
    BoundProgram,
    BoundReturnStmt,
    BoundStmt,
    BoundVariable,
    BoundWhileStmt,
)


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

class Return(MStmt):
    val: Operand | RValue

    def __init__(self, val: Operand | RValue) -> None:
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


class Block:
    id: int
    stmts: list[MStmt]
    term: TermStmt
    preds: list["Block"]

    def __init__(self, id: int) -> None:
        self.id = id
        self.stmts = []
        self.term = Halt([])
        self.preds = []

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

    def __init__(self, fn: BoundFunction, mir_context: "MIRContext") -> None:
        self.tmp_counter = 0
        self.block_counter = 0
        self.fn = fn
        self.locals = {}
        self.local_counter = len(self.locals)
        self.blocks = []
        self.mir_context = mir_context

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
            return self.mir_context.get_global(sym)
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
                phi = Phi([(true_block.id, rhs), (false_block.id, Const(False))])
                self.emit(Assign(tmp, phi))
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
            self.emit(Return(val))
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
            self.current_block().term = Goto(block_id=exit_block.id, succ=[exit_block])
            self.blocks.append(false_block)
            for s in stmt.else_block.stmts:
                self.lower_stmt(s)
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
            for s in stmt.block.stmts:
                self.lower_stmt(s)
            self.current_block().term = Goto(succ=[head_block], block_id=head_block.id)
            self.blocks.append(exit_block)
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

    def get_global(self, sym: Symbol):
        if sym.sym_type == SymbolType.Global:
            local = self.globals[sym.symbol_id]
            assert local is not None, "Local was none in get local"
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


class PhiAssign(MStmt):
    id: int
    phi: Phi
    def __init__(self, id: int, phi: Phi):
        self.phi = phi
        self.id = id

    @override
    def __repr__(self):
        return f"φ{self.id} = {self.phi}"

class PhiBlock:
    id: int
    stmts: list[MStmt]
    term: TermStmt
    preds: list["Block"]
    phis: dict[int, PhiAssign]

    def __init__(self, id: int) -> None:
        self.id = id
        self.stmts = []
        self.term = Halt([])
        self.preds = []
        self.phis = {}

    @override
    def __repr__(self) -> str:
        return f"Block({self.id})"

class SSANamer:
    df: dict[int, set[Block]]
    cfg: CFGBuilder
    vars: dict[int, list[int]]
    var_counter: int
    def_blocks: dict[int, set[PhiBlock]]

    def __init__(self, df: dict[int, set[Block]], cfg: CFGBuilder) -> None:
        self.df = df
        self.cfg = cfg
        self.vars = defaultdict(list)
        self.var_counter = 0
        self.def_blocks = defaultdict(set)

    def new_name(self):
        new = self.var_counter
        self.var_counter += 1
        return new

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

    def topvar(self, id: int):
        assert len(self.vars[id]) > 0
        return self.vars[id][-1]

    def rename_place(self, place: Place) -> Place:
        match place:
            case Temp(id=id):
                return Temp(self.topvar(id))
            case Global(id=id, sym=sym):
                return Global(self.topvar(sym.symbol_id), sym)
            case Local(id=id, sym=sym):
                return Local(self.topvar(sym.symbol_id), sym)
            case _:
                raise AssertionError(f"unhandled place type in renamer: {type(place)}")

    def define_place(self, place: Place) -> Place:
        match place:
            case Temp(id=id):
                self.vars[id].append(len(self.vars[id]))
                return Temp(len(self.vars[id])-1)
            case Global(id=id, sym=sym):
                self.vars[sym.symbol_id].append(len(self.vars[sym.symbol_id]))
                return Global(len(self.vars[sym.symbol_id])-1, sym)
            case Local(id=id, sym=sym):
                self.vars[sym.symbol_id].append(len(self.vars[sym.symbol_id]))
                return Local(len(self.vars[sym.symbol_id])-1, sym)
            case _:
                raise AssertionError(f"unhandled place type in renamer: {type(place)}")

    def rename_stmt(self, stmt: MStmt):
        match stmt:
            case Assign(lval=lval, rval=rval):
                new_rval = self.rename_rval(rval)
                new_lval = self.define_place(lval)
                return Assign(new_lval, new_rval)
            case Return(val=val):
                new_val = self.rename_rval(val)
                return Return(new_val)
            case _:
                raise AssertionError(f"unhandled stmt kind in ssa renaming: {type(stmt)}")


    def var_key_from_place(self, place: Place) -> int:
        match place:
            case Temp(id=id):
                return id
            case Global(id=_, sym=sym):
                return sym.symbol_id
            case Local(id=_, sym=sym):
                return sym.symbol_id
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

    def run(self):
        print("<<<<<<<<RENAMING>>>>>>>>")
        map: dict[int, int] = {}
        blocks: list[PhiBlock] = []
        block_defines_var: dict[int, set[int]] = defaultdict(set)
        for idx, block in enumerate(self.cfg.blocks):
            new_block = PhiBlock(block.id)
            new_block.preds = block.preds
            new_block.term = block.term
            new_block.stmts = block.stmts
            for stmt in new_block.stmts:
                match stmt:
                    case Assign(lval=lval, rval=rval):
                        k = self.var_key_from_place(lval)
                        self.def_blocks[k].add(new_block)
                        block_defines_var[k].add(new_block.id)
                    case _:
                        pass
            map[block.id] = idx
            blocks.append(new_block)
        self.def_blocks = defaultdict(set, {k: v for (k, v) in self.def_blocks.items() if len({b.id for b in v}) >= 2})
        block_defines_var = defaultdict(set, {k: v for (k, v) in block_defines_var.items() if len(v) >= 2})
        hasphi: dict[int, set[int]] = defaultdict(set)                 # var -> {join_id}
        phi_sources: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

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
                            blocks[j_idx].phis[var] = blocks[j_idx].phis.get(var, PhiAssign(var, Phi([])))
                            hasphi[var].add(j_id)
                        if j_id not in visited:
                            phi_sources[var][j_id].add(source)

                        if j_id not in block_defines_var[var] and j_id not in visited:
                            visited.add(j_id)
                            stack.append(blocks[j_idx])

        for block in blocks:
            if not block.phis:
                continue
            for var_key, phi_assign in block.phis.items():
                srcs: set[int] = set()
                for pred in block.preds:
                    src = self.nearest_def_block_id(pred, var_key, block_defines_var)
                    if src is not None:
                        srcs.add(src)
                phi_assign.phi.incoming = sorted(srcs)

        for block in blocks:
            new_stmts: list[MStmt] = []
            for stmt in block.stmts:
                new_stmts.append(self.rename_stmt(stmt))
            block.stmts = new_stmts

        for block in blocks:
            for stmt in block.stmts:
                print(stmt)
        print("===PHIS===")
        for block in blocks:
            for (id, assign) in block.phis.items():
                print(assign)
        print("===END PHIS===")
        print("<<<<<<<<RENAMING FINISHED>>>>>>>>")


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
                dominator = tree.map[block.id]
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
        for (cfg, df) in zip(self.cfgs, self.DFs):
            namer = SSANamer(df=df, cfg=cfg)
            namer.run()

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
let a = 1+1;
let b = 2+2;
let c = 4+2;
fn test() {
    print("test");
    let a = 0;
    while a < 10 {
        a += 1;
    }
    return a;
}
test();
a = 2;
print(a+1);
while true {

}
if (a == 2) {
    print("true");
    if (a ==2) {
        a = 1;
        print("true");
    } else {
        a = 2;
        print("false");
    }
} else {
    print("false");
}
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

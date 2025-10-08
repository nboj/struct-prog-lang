from dataclasses import dataclass

from .tokenizer import Token, TokenType
from .types import NIL, Nil, Op, Symbol, SymbolType
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


@dataclass(frozen=True)
class MStmt:
    pass


@dataclass(frozen=True)
class TermStmt:
    succ: list["Block"]


@dataclass(frozen=True)
class Branch(TermStmt):
    condition: "Operand"
    true_block_id: int
    false_block_id: int
    succ: list["Block"]

@dataclass(frozen=True)
class Goto(TermStmt):
    block_id: int
    succ: list["Block"]

@dataclass(frozen=True)
class Halt(TermStmt):
    succ: list["Block"]

@dataclass(frozen=True)
class Place:
    pass


@dataclass(frozen=True)
class Local(Place):
    id: int
    sym: Symbol


@dataclass(frozen=True)
class Global(Place):
    id: int
    sym: Symbol


@dataclass(frozen=True)
class RValue:
    pass


@dataclass(frozen=True)
class Const(RValue):
    value: int | float | str | bool | Nil

@dataclass(frozen=True)
class Temp(Place):
    id: int


@dataclass(frozen=True)
class Copy:
    origin: Place


@dataclass(frozen=True)
class Move:
    origin: Place


type Operand = Const | Copy | Move


@dataclass(frozen=True)
class BinOp(RValue):
    op: Op
    lhs: Operand
    rhs: Operand


@dataclass(frozen=True)
class DirectFunc:
    func_id: int
    func_type: SymbolType


@dataclass(frozen=True)
class IndirectFunc:
    operand: Operand
    func_type: SymbolType


@dataclass(frozen=True)
class Call(RValue):
    callee: DirectFunc | IndirectFunc
    args: list[Operand]


@dataclass(frozen=True)
class Assign(MStmt):
    lval: Place
    rval: RValue | Operand

@dataclass(frozen=True)
class Return(MStmt):
    val: Operand | RValue

@dataclass(frozen=True)
class Phi(RValue):
    incoming: list[tuple[int, Operand]] # (pred_block_id, value)


class Block:
    id: int
    stmts: list[MStmt]
    term: TermStmt
    preds: list["Block"]

    def __init__(self, id) -> None:
        self.id = id
        self.stmts = []
        self.term = Halt([])
        self.preds = []


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
                return DirectFunc(callee.sym.symbol_id, sym.sym_type)
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
                    print(expr.op.kind)
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
    block_ref: Block | None
    children: list["DominatorNode"]

    def __init__(self, block_ref: Block | None, parent: "DominatorNode | None" = None) -> None:
        self.parent = parent
        self.children = []
        self.block_ref = block_ref

class DomTreeBuilder:
    cfg: CFGBuilder
    entry: DominatorNode

    def __init__(self, cfg: CFGBuilder) -> None:
        self.cfg = cfg
        self.entry = DominatorNode(None)

    def get_dominators(self, node: Block):
        dominators = []
        for pred in node.preds:
            if pred is self.entry:
                dominators.append(self.entry)
                return dominators

    def dominates(self, node: Block, other: Block):
        for pred in node.preds:
            pass

    def build(self):
        if len(self.cfg.blocks) <= 0:
            return
        map: dict[int, Block] = {}
        self.entry.block_ref = self.cfg.blocks[0]
        for block in self.cfg.blocks:
            map[block.id] = block

        dominator_nodes = {self.entry.block_ref.id: self.entry}

        def get_dominator(block:Block):
            if len(block.preds) == 1:
                dom_node = dominator_nodes.get(block.preds[0].id)
                if not dom_node:
                    dom_node = DominatorNode(block.preds[0])
                    dominator_nodes[block.preds[0].id] = dom_node
                return dom_node
            elif len(block.preds) == 0:
                dom_node = self.entry
                return dom_node
            else:
                preds = block.preds
                while True:
                    doms = []
                    for node in preds:
                        dominator = get_dominator(node)
                        doms.append(dominator)
                    prev = None
                    same = True
                    for dom in doms:
                        if prev is None:
                            prev = dom
                        elif prev != dom:
                            same = False
                            break
                    new_preds = []
                    if same == True:
                        return dominator_nodes.get(preds[0].id)
                    else:
                        for pred in preds:
                            new_preds += pred.preds
                        preds = new_preds

        def compute_node(block: Block):
            node = DominatorNode(block)
            dominator = get_dominator(block)
            assert dominator is not None
            node.parent = dominator
            dominator.children.append(node)
            #if dominator_nodes.get(block.id) is None:
            dominator_nodes[block.id] = node
            return node

        def compute_tree(current_node: Block):
            root = compute_node(current_node)
            for succ in current_node.term.succ:
                _ = compute_tree(succ)
            return root

        self.entry = compute_tree(self.entry.block_ref)

        def print_node(node: DominatorNode):
            assert node.block_ref is not None
            assert node.parent is not None
            assert node.parent.block_ref is not None
            print()
            print("BLOCKID:", node.block_ref.id)
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

        def print_tree(node: DominatorNode):
            print_node(node)
            for child in node.children:
                print_node(child)
        print_tree(self.entry);



class Mir:
    program: BoundProgram
    temp_counter: int
    debug: bool
    cfgs: list[CFGBuilder]
    dom_trees: list[DomTreeBuilder]
    context: MIRContext

    def __init__(self, bound_program: BoundProgram, debug: bool = False) -> None:
        self.program = bound_program
        self.temp_counter = 0
        self.debug = debug
        self.cfgs = []
        self.dom_trees = []
        self.context =  MIRContext()

    def build_cfg(self):
        for fn in self.program.body:
            if self.debug:
                print(f"NEW FUNCTION: {fn.sym.name}")
            builder = CFGBuilder(fn, self.context)
            builder.build()
            self.cfgs.append(builder)

    def build_trees(self):
        for cfg in self.cfgs:
            tree = DomTreeBuilder(cfg)
            tree.build()
            self.dom_trees.append(tree)

    def lower(self):
        self.build_cfg()
        self.build_trees()


if __name__ == "__main__":
    from .tokenizer import Tokenizer
    from .parser import Parser
    from .binder import Binder

    text = """
fn test() {
    print("test");
    let a = 0;
    if a == 0 {
        a += 1;
    }
    return a;
}
let a = 1+1;
a = 2;
print(a+1);
if (a == 2 && a == 2) {
    print("true");
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

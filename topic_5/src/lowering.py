from .source_map import SourceMap
from .binder import *
from .types import NIL, CodeObject, Instr, ModuleLayout, Nil, Op, PseudoOp, SlotLayout
from .tokenizer import TokenType

class Lowering:
    sm: SourceMap
    program: BoundProgram
    label_map: dict[str, int]
    bytecode: list[Instr]
    pseudo_bytecode: list[Instr]
    consts: list[object]
    const_index: dict[object, int]
    label_counter: int
    fn_stack: list[SlotLayout]
    modules: list[ModuleLayout]
    nil_idx: int
    loop_stack: list[tuple[int, int]] # NOTE: (start, end)

    def __init__(self, program: BoundProgram, source_map: SourceMap):
        self.sm = source_map
        self.program = program
        self.label_map = {}
        self.bytecode = []
        self.pseudo_bytecode = []
        self.consts = []

        self.const_index = {}
        self.nil_idx = self.intern_const(NIL)
        self.true_idx = self.intern_const(True)
        self.false_idx = self.intern_const(False)

        self.label_counter = 0
        self.fn_stack = []
        self.modules = [ModuleLayout("main", 0)]
        self.loop_stack = []

    def intern_const(self, const: int | str | bool | float | Nil):
        key = (type(const), const) # NOTE: this is a python method to prevent collisions
        index = self.const_index.get(key)
        if index is None:
            self.consts.append(const)
            index = len(self.consts) - 1
            self.const_index[key] = index
        return index

    def get_main_module(self):
        return self.modules[0]

    def get_current_fn_layout(self):
        assert len(self.fn_stack) > 0, "tried accessing a null fn layout in lowering.py"
        return self.fn_stack[len(self.fn_stack)-1]

    def emit_pseudo_nil(self):
        self.emit_pseudo(Op.PUSHK, self.nil_idx)

    def lower_expression(self, expr: BoundExpr, want_value=True):
        if isinstance(expr, BoundLiteral):
            index = self.intern_const(expr.value)
            if not want_value:
                return
            self.emit_pseudo(Op.PUSHK, index)
        elif isinstance(expr, BoundVariable):
            if not want_value:
                return
            sym = expr.sym
            if sym.sym_type == SymbolType.Global:
                slot = self.get_main_module().get_global(sym)
                self.emit_pseudo(Op.LOADG, slot)
            elif sym.sym_type == SymbolType.Local:
                slot = self.get_current_fn_layout().get_slot(sym)
                self.emit_pseudo(Op.LOAD, slot)
            else:
                raise AssertionError(f"unhandled BoundVariable SymbolType: {sym.sym_type}")
        elif isinstance(expr, BoundAssign):
            if isinstance(expr.target, BoundVariable): 
                self.lower_expression(expr.value)
                target_sym = expr.target.sym
                if target_sym.sym_type == SymbolType.Global:
                    target_slot = self.get_main_module().get_global(target_sym)
                    self.emit_pseudo(Op.STOREG, target_slot)
                    if not want_value:
                        return
                    self.emit_pseudo_nil()
                elif target_sym.sym_type == SymbolType.Local:
                    target_slot = self.get_current_fn_layout().get_slot(target_sym)
                    self.emit_pseudo(Op.STORE, target_slot)
                    if not want_value:
                        return
                    self.emit_pseudo_nil()
                else:
                    raise AssertionError(f"unhandled SymbolType in lowering.py: {target_sym.sym_type}")
            else:
                raise AssertionError(
                    self.sm.to_err(
                        expr, "unhandled BoundExpr in lowering.py BoundAssign"
                    )
                )
        elif isinstance(expr, BoundCallExpr):
            if isinstance(expr.callee, BoundVariable):
                sym = expr.callee.sym
                if sym.sym_type == SymbolType.Builtin:
                    for arg in expr.args:
                        self.lower_expression(arg)
                    self.emit_pseudo(Op.CALL_BUILTIN, sym.symbol_id, len(expr.args))
                    if not want_value:
                        self.emit_pseudo(Op.POPN, 1)
                else:
                    raise AssertionError("custom functions not implemented in lowering.py")
            else:
                raise AssertionError(f"unhandled callee expr: {expr}")
        elif isinstance(expr, BoundBinary):
            left = expr.left
            right = expr.right
            if not want_value:
                if expr.op.kind in (TokenType.DoubleAmp, TokenType.DoublePipe):
                    self.lower_expression(expr)
                    self.emit_pseudo(Op.POPN, 1)
                    return
                else:
                    self.lower_expression(left, want_value=False)
                    self.lower_expression(right, want_value=False)
                    return
            match expr.op.kind:
                case TokenType.Plus:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.ADD)
                case TokenType.Minus:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.SUB)
                case TokenType.Divide:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.DIV)
                case TokenType.Star:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.MUL)
                case TokenType.DbEq:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.EQ)
                case TokenType.NEq:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.NEQ)
                case TokenType.Lt:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.LT)
                case TokenType.Gt:
                    self.lower_expression(left)
                    self.lower_expression(right)
                    self.emit_pseudo(Op.GT)
                case TokenType.DoublePipe:
                    endL = self.new_label()
                    rhsL = self.new_label()
                    falseL = self.new_label()
                    self.lower_expression(left)
                    self.emit_pseudo(PseudoOp.JZ_LABEL, rhsL)
                    self.emit_pseudo(Op.PUSHK, self.true_idx)
                    self.emit_pseudo(PseudoOp.JMP_LABEL, endL)
                    self.emit_pseudo(PseudoOp.LABEL, rhsL)

                    self.lower_expression(right)
                    self.emit_pseudo(PseudoOp.JZ_LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.true_idx)
                    self.emit_pseudo(PseudoOp.JMP_LABEL, endL)
                    self.emit_pseudo(PseudoOp.LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.false_idx)

                    self.emit_pseudo(PseudoOp.LABEL, endL)
                    #self.emit_pseudo(Op.OR)
                case TokenType.DoubleAmp:
                    endL = self.new_label()
                    falseL = self.new_label()
                    self.lower_expression(left)
                    self.emit_pseudo(PseudoOp.JZ_LABEL, falseL)
                    self.lower_expression(right)
                    self.emit_pseudo(PseudoOp.JZ_LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.true_idx)
                    self.emit_pseudo(PseudoOp.JMP_LABEL, endL)
                    self.emit_pseudo(PseudoOp.LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.false_idx)
                    self.emit_pseudo(PseudoOp.LABEL, endL)
                    #self.emit_pseudo(Op.AND)
                case _:
                    raise AssertionError(f"unhandled op kind in lowering.py: {expr.op.kind}")
        elif isinstance(expr, BoundUnary):
            if not want_value:
                self.lower_expression(expr.expr, False)
                return
            match expr.op.kind:
                case TokenType.Minus:
                    if isinstance(expr.expr, BoundLiteral):
                        if isinstance(expr.expr.value, bool):
                            raise AssertionError(self.sm.to_err(expr.expr, "unary - not supported for bool"))
                        if not isinstance(expr.expr.value, (int, float)):
                            raise AssertionError(self.sm.to_err(expr.expr, f"unary - not supported for {type(expr.expr.value).__name__}"))
                        index = self.intern_const(-expr.expr.value)
                        self.emit_pseudo(Op.PUSHK, index)
                        return
                    self.lower_expression(expr.expr)
                    self.emit_pseudo(Op.NEG)
                case TokenType.Bang:
                    endL = self.new_label()
                    falseL = self.new_label()
                    self.lower_expression(expr.expr)
                    self.emit_pseudo(PseudoOp.JZ_LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.false_idx)
                    self.emit_pseudo(PseudoOp.JMP_LABEL, endL)

                    self.emit_pseudo(PseudoOp.LABEL, falseL)
                    self.emit_pseudo(Op.PUSHK, self.true_idx)

                    self.emit_pseudo(PseudoOp.LABEL, endL)
                    #self.emit_pseudo(Op.NOT)
                case _:
                    raise AssertionError(self.sm.to_err(expr, f"unhandled expr type in lower_expression in lowering.py: {expr}"))
        else:
            raise AssertionError(
                self.sm.to_err(expr, f"unhandled BoundExpr in lowering.py: {expr}")
            )

    def emit_pseudo(
        self, op: Op | PseudoOp, a: int | None = None, b: int | None = None
    ):
        self.pseudo_bytecode.append(Instr(op, a, b))

    def new_label(self) -> int:
        label = self.label_counter
        self.label_counter += 1
        return label

    def lower_block(self, block: BoundBlockStmt):
        for stmt in block.stmts:
            self.lower_stmt(stmt)

    def enter_loop(self, startL: int, endL: int):
        self.loop_stack.append((startL, endL))

    def exit_loop(self):
        _ = self.loop_stack.pop()

    def lower_stmt(self, stmt: BoundStmt):
        if isinstance(stmt, BoundIfStmt):
            elseL = self.new_label()
            endL = self.new_label()

            self.lower_expression(stmt.condition)
            self.emit_pseudo(PseudoOp.JZ_LABEL, elseL)
            self.lower_block(stmt.then_block)
            self.emit_pseudo(PseudoOp.JMP_LABEL, endL)

            self.emit_pseudo(PseudoOp.LABEL, elseL)
            self.lower_block(stmt.else_block)
            self.emit_pseudo(PseudoOp.LABEL, endL)
        elif isinstance(stmt, BoundExprStmt):
            self.lower_expression(stmt.expr, False)
            #self.emit_pseudo(Op.POPN, 1)
        elif isinstance(stmt, BoundLetStmt):
            assert isinstance(stmt.assign.target, BoundVariable)
            self.lower_expression(stmt.assign.value)
            sym = stmt.assign.target.sym
            if sym.sym_type == SymbolType.Global:
                slot = self.get_main_module().ensure_global(sym)
                self.emit_pseudo(Op.STOREG, slot)
            elif sym.sym_type == SymbolType.Local:
                slot = self.get_current_fn_layout().ensure_slot(sym)
                self.emit_pseudo(Op.STORE, slot)
            else:
                raise AssertionError(self.sm.to_err(stmt, f"unhandled SymbolType in let statement: {sym.sym_type}"))
        elif isinstance(stmt, BoundWhileStmt):
            startL = self.new_label()
            endL = self.new_label()
            self.enter_loop(startL, endL)
            self.emit_pseudo(PseudoOp.LABEL, startL)
            self.lower_expression(stmt.condition)
            self.emit_pseudo(PseudoOp.JZ_LABEL, endL)
            self.lower_block(stmt.block)
            self.emit_pseudo(PseudoOp.JMP_LABEL, startL)
            self.emit_pseudo(PseudoOp.LABEL, endL)
            self.exit_loop()
        elif isinstance(stmt, BoundBreakStmt):
            self.emit_pseudo(PseudoOp.JMP_LABEL, self.loop_stack[len(self.loop_stack)-1][1])
        elif isinstance(stmt, BoundContinueStmt):
            self.emit_pseudo(PseudoOp.JMP_LABEL, self.loop_stack[len(self.loop_stack)-1][0])
        elif isinstance(stmt, BoundBlockStmt):
            self.lower_block(stmt)
        else:
            raise AssertionError(
                self.sm.to_err(stmt, f"unhandled BoundStmt in lowering.py: {stmt}")
            )

    def lower(self):
        for stmt in self.program.body:
            self.lower_stmt(stmt)

        map: dict[int, int] = {} # NOTE: id, location
        pc: int = 0
        for instr in self.pseudo_bytecode:
            if isinstance(instr.op, PseudoOp):
                match instr.op:
                    case PseudoOp.LABEL:
                        assert instr.a is not None, "LABEL attr: a was None"
                        map[instr.a] = pc
                    case _:
                        pc += 1
                        # raise AssertionError(f"unhandled pseudo op instruction in lowering.py: {instr.op_str(instr.op)}")
            elif isinstance(instr.op, Op):
                pc += 1
            else:
                raise AssertionError(f"unexpected instr op type in lowering.py: {instr.op}")

        for instr in self.pseudo_bytecode:
            if isinstance(instr.op, PseudoOp):
                match instr.op:
                    case PseudoOp.JZ_LABEL:
                        assert instr.a is not None, "JZ_LABEL attr: a was None"
                        #self.emit(Op.JZ, map[instr.a])
                        self.bytecode.append(Instr(Op.JZ, map[instr.a]))
                    case PseudoOp.LABEL:
                        pass
                    case PseudoOp.JMP_LABEL:
                        assert instr.a is not None, "JMP_LABEL attr: a was None"
                        #self.emit(Op.JMP, map[instr.a])
                        self.bytecode.append(Instr(Op.JMP, map[instr.a]))
            else:
                self.bytecode.append(instr)

        self.bytecode.append(Instr(Op.HALT))
        return CodeObject(self.bytecode, self.consts, 0, len(self.get_main_module().layout.map))

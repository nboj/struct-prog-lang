from typing import Any
from ..src.utils import compile_trivial
from ..src.parser import Node, Parser, Program, ExprStmt, CallExpr, Span, Variable, Binary, Literal
from ..src.tokenizer import Tokenizer, Token, TokenType
from ..src.debug_ast import render_ast


def test_call():
    tokenizer = Tokenizer("print(1+1);")
    parser = Parser(tokenizer.tokenize(), tokenizer.sm)
    parsed = parser.parse()
    assert isinstance(parsed, Program)
    assert isinstance(parsed.body[0], ExprStmt)
    assert isinstance(parsed.body[0].expr, CallExpr)
    assert isinstance(parsed.body[0].expr.callee, Variable)
    assert isinstance(parsed.body[0].expr.args[0], Binary)
    print("Call test passed")


def test_syntax():
    try:
        tokenizer = Tokenizer("print(1++1);")
        parser = Parser(tokenizer.tokenize(), tokenizer.sm)
        parser.parse()
    except Exception:
        print("Call test passed")
        pass


def test_syntax2():
    try:
        tokenizer = Tokenizer("print();")
        parser = Parser(tokenizer.tokenize(), tokenizer.sm)
        parsed = parser.parse()
        assert isinstance(parsed, Program)
        assert isinstance(parsed.body[0], ExprStmt)
        assert isinstance(parsed.body[0].expr, CallExpr)
        assert isinstance(parsed.body[0].expr.callee, Variable)
        print("Call test passed")
    except Exception:
        assert False, "Test Syntax2 failed."

def test_parse_factor():
    """
    factor = <int> | <number> | <bool> | <ident> | "(" expression ")"
    """
    def verify_is_literal(parsed: Program, lit: Any):
        assert isinstance(parsed, Program)
        assert isinstance(parsed.body[0], ExprStmt)
        assert isinstance(parsed.body[0].expr, Literal)
        assert parsed.body[0].expr.value == lit
        assert len(parsed.body) == 1

    vals = [
        328,
        "this is a string",
        "234 234389",
        "tn 23n3t 22398t 23",
        "",
        2.,
        .2,
        0.2,
        2.0382824,
        True,
        False,
    ]
    tests = []
    for val in vals:
        if isinstance(val, str):
            tests.append(f'"{val}";')
        else:
            tests.append(f"{val};")

    for test, val in zip(tests, vals):
        tokenizer = Tokenizer(test)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens, tokenizer.sm)
        parsed = parser.parse()
        print(parsed)
        verify_is_literal(parsed, val)

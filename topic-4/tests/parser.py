from src.parser import Parser, Program, ExprStmt, CallExpr, Span, Variable, Binary, Literal
from src.tokenizer import Tokenizer, Token, TokenType
from src.debug_ast import render_ast


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

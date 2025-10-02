import random
import re
from dataclasses import dataclass
from typing import Union

from pytest import approx

from .utils import compile_trivial, run_vm, parse_optional

Number = Union[int, float]

FUZZ_SEED = 1337
NUM_EXPRESSIONS = 200
MAX_DEPTH = 4
ALWAYS_PAREN = True  # FIXME: Figure out why setting to False breaks it

VARS = (
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
    "aa bb cc dd ee ff gg hh ii jj kk"
    "aaa bbl ccll ddl ele ffl gllg hllh illi jllj kllk"
).split()


def make_declare(vals: dict[str, int]) -> str:
    lines = []
    for name in sorted(vals.keys(), key=lambda s: (len(s), s)):
        lines.append(f"let {name} = {vals[name]};")
    return "\n".join(lines) + "\n"


def parse_decl(declare: str) -> dict[str, int]:
    vals = {}
    for name, val in re.findall(r"let\s+([A-Za-z_]\w*)\s*=\s*([0-9_]+)\s*;", declare):
        vals[name] = int(val.replace("_", ""))
    return vals


def rand_nonzero_int(lo=1, hi=200) -> int:
    x = 0
    while x == 0:
        x = random.randint(lo, hi)
    return x


def paren(s: str) -> str:
    return f"({s})"


@dataclass
class Built:
    lang: str
    py: str
    val: Union[Number, bool]


def build_number(vals: dict[str, int], depth: int) -> Built:
    """Build numeric expression (+,-,*,/ with no div-by-zero)."""
    if depth <= 0 or random.random() < 0.25:
        if random.random() < 0.8:
            v = random.choice(list(vals.keys()))
            return Built(v, v, vals[v])
        lit = (
            rand_nonzero_int(1, 25)
            if random.random() < 0.9
            else random.randint(-25, 25) or 1
        )
        return Built(str(lit), str(lit), lit)

    left = build_number(vals, depth - 1)
    right = build_number(vals, depth - 1)
    op = random.choice(["+", "-", "*", "/"])
    if op == "/":
        retries = 0
        while isinstance(right.val, (int, float)) and right.val == 0 and retries < 10:
            right = build_number(vals, depth - 1)
            retries += 1
        if isinstance(right.val, (int, float)) and right.val == 0:
            right = Built("1", "1", 1)

    if ALWAYS_PAREN or random.random() < 0.7:
        lang = f"{paren(left.lang)} {op} {paren(right.lang)}"
        py = f"{paren(left.py)} {op} {paren(right.py)}"
    else:
        lang = f"{left.lang} {op} {right.lang}"
        py = f"{left.py} {op} {right.py}"

    if op == "+":
        val = float(left.val) + float(right.val)
    elif op == "-":
        val = float(left.val) - float(right.val)
    elif op == "*":
        val = float(left.val) * float(right.val)
    else:
        val = float(left.val) / float(right.val)

    if isinstance(val, float) and val.is_integer():
        val = int(val)

    return Built(lang, py, val)


def build_compare(vals: dict[str, int], depth: int) -> Built:
    """Build a comparison (> < ==) of numeric expressions; fully parenthesized to avoid Python's chaining."""
    op = random.choice([">", "<", "=="])
    a = build_number(vals, max(0, depth - 1))
    b = build_number(vals, max(0, depth - 1))
    lang = f"{paren(a.lang)} {op} {paren(b.lang)}"
    py = f"{paren(a.py)} {op} {paren(b.py)}"
    if op == ">":
        val = a.val > b.val
    elif op == "<":
        val = a.val < b.val
    else:
        val = a.val == b.val
    return Built(lang, py, val)


def build_bool(vals: dict[str, int], depth: int) -> Built:
    if depth <= 0 or random.random() < 0.35:
        return build_compare(vals, 1)

    left = build_bool(vals, depth - 1)
    right = build_bool(vals, depth - 1)
    op = random.choice(["&&", "||"])
    if ALWAYS_PAREN or random.random() < 0.8:
        lang = f"({left.lang}) {op} ({right.lang})"
    else:
        lang = f"{left.lang} {op} {right.lang}"

    py_op = "and" if op == "&&" else "or"
    py = f"({left.py}) {py_op} ({right.py})"
    if op == "&&":
        val = bool(left.val and right.val)
    else:
        val = bool(left.val or right.val)

    return Built(lang, py, val)


def build_expr(vals: dict[str, int], depth: int) -> Built:
    """Randomly pick a number or boolean expression."""
    if random.random() < 0.5:
        return build_number(vals, depth)
    else:
        return build_bool(vals, depth)


# CREDIT: GPT
def test_fuzz_expressions():
    random.seed(FUZZ_SEED)

    # Create a consistent declaration (non-zero by default)
    vals = {name: rand_nonzero_int(1, 200) for name in VARS}
    declare = make_declare(vals)

    # Generate program & expected values
    expected_vals = []
    lines = ["// --- fuzz prints ---"]
    for _ in range(NUM_EXPRESSIONS):
        e = build_expr(vals, MAX_DEPTH)
        lines.append(f"print({e.lang});")
        expected_vals.append(e.val)

    program = "\n".join(lines) + "\n"
    full_src = declare + program

    codeobj = compile_trivial(full_src)
    # optional: pretty_ir(codeobj.code)
    out, vm = run_vm(codeobj)
    outputs = [parse_optional(s) for s in out.splitlines() if s.strip()]
    for line in lines:
        print(line)
    print()
    for o in out.splitlines():
        print(out)
    print(vals)

    # Sanity
    assert vm.sp == 0
    assert len(outputs) == len(expected_vals)

    # Compare with tolerance for numbers; exact for bools
    for got, exp in zip(outputs, expected_vals):
        if isinstance(exp, bool):
            # VM may return "true"/"false" parsed by parse_optional -> bool
            assert isinstance(got, (bool, int, float, str))
            # If got is a number/string, coerce to bool in the same way your VM prints booleans.
            if isinstance(got, bool):
                assert got is exp
            else:
                # Accept "true"/"false" strings or 1/0 if your VM does that
                normalized = str(got).strip().lower()
                assert normalized in ("true", "false", "1", "0")
                assert (normalized in ("true", "1")) is exp
        else:
            # numeric
            assert isinstance(got, (int, float, str))
            if isinstance(got, (int, float)):
                assert got == approx(exp, rel=1e-12, abs=1e-12)
            else:
                # String path (VM prints as text). Try parsing to float/int here if your parse_optional missed it.
                parsed = parse_optional(got)
                if isinstance(parsed, (int, float)):
                    assert parsed == approx(exp, rel=1e-12, abs=1e-12)
                else:
                    # Last resort: compare as string with Python-evaluated formatted value
                    assert got == f"{exp}"

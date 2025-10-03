from ..src.types import Instr, Op
from ..src.utils import compile_trivial, parse_optional, pretty_ir, run_vm
from pytest import approx


def test_expressions():
    codeobj = compile_trivial(
        """
1+1;
"""
    )
    code = codeobj.code
    pretty_ir(code)
    assert code == [
        Instr(Op.PUSHK, 1),
        Instr(Op.PUSHK, 1),
        Instr(Op.ADD),
        Instr(Op.POPN, 1),
        Instr(Op.HALT),
    ]


def test_expressions2():
    codeobj = compile_trivial(
        """
print(3*1+1);
"""
    )
    code = codeobj.code
    pretty_ir(code)
    truth = [
        Instr(Op.PUSHK, 1),
        Instr(Op.PUSHK, 2),
        Instr(Op.MUL),
        Instr(Op.PUSHK, 2),
        Instr(Op.ADD),
        Instr(Op.CALL_BUILTIN, 0, 1),
        Instr(Op.POPN, 1),
        Instr(Op.HALT),
    ]
    print()
    pretty_ir(truth)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    assert code == truth
    assert vm.sp == 0
    assert len(lines) == 1
    assert lines[0] == "4"


def test_expressions3():
    codeobj = compile_trivial(
        """
let a = 32;
let b = 323;
let c = 2;
a = a + 5;
b = b * 5;
print(3*1+1+a*b/c);
"""
    )
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert lines[0] == "29881.5"
    assert vm.sp == 0


def test_expressions4():
    base1 = """
print(a + b * c - d / e);
print(x*y + z/w - p*q + r/s);
print(a*b*c*d + e/f - g*h/i);
"""
    base2 = """
print(a + b * c > d - e / f);
print((a + b) * c < d / (e - f));
print((a*b + c/d) == (e*f - g/h));
print(u+v*w == x+y/z);
print(m*n - p/q > r*s + t/u);
print((a+b)/(c+d) < (e+f)/(g+h));
"""


    # Logical mixes (no extra parens)
    base3 = """
print(a + b > c && d < e);
print(a * b == c || d / e > f);
print(a + b * c == d && e < f * g);
print(x / y < z || p * q == r - s);
print(a + b > c || d + e < f && g == h);
"""

    # Parentheses to force grouping
    base4 = """
print(((a + b) * (c - d)) / ((e + f) * (g - h)));
print((((a + b) * (c - d)) / (e + f)) > ((g * h) - (i / j)));
print(((a*b) + (c/d) - (e*f)) == ((g/h) + (i*j) - (k/l)));
print((((a+b)+(c+d))*((e+f)-(g+h))) < ((i*j)/(k+l)));
"""

    # Deep nesting (stress parser/stack)
    base5 = """
print(((((((a + b) * (c + d)) - (e / f)) + (g * (h + i))) / (j + (k * l))) == (m + (n * (o + p)))));
print(((((a+b)-((c*d)+(e/f))) * ((g+h)/(i+j))) > (((k*l)-(m/n))+((o+p)-(q/r)))));
"""

    # Long logical chains for associativity
    base6 = """
print((a + b > c) && (d + e > f) && (g + h > i) && (j + k > l));
print((a*b == c*d) || (e/f < g/h) || (i + j > k - l) || (m*n == o/p));
print((a < b) && (b < c) && (c < d) && (d < e) && (e < f) && (f < g));
print((a == b) || (b == c) || (c == d) || (d == e) || (e == f));
"""

    # Mixed precedence without redundant parens
    base7 = """
print(a + b > c && d + e == f || g * h < i);
print(a*b + c == d && e/f < g || h - i > j);
print(x + y * z < p && q == r + s || t / u > v);
"""

    # Over-parenthesized (parser should still accept)
    base8 = """
print((((((a + b)))) * (((c - d)))) == (((e * f))));
print((((a+b)*(c+d)) - ((e+f)/(g+h))) > (((i+j) * (k+l)) - (m+n)));
"""
    # Newline and whitespace stress
    base9 = """
print(a   +    b*c   -   d/  e);
print(a+b*c -d/e +f*g);
print((a+b) && (c<d) || (e==f));
print(x*y / ( a + b + c + d + e ));
"""

    # Very long expression (token stream stress)
    base10 = """
print(a+b*c - d/e + f*g - h/i + j*k - l/m + n*o - p/q + r*s - t/u + v*w - x/y + z*aa - bb/cc + dd*ee - ff/gg + hh*ii - jj/kk);
"""
    # Equality and comparisons chained with logic (parenthesized comparisons)
    base11 = """
print(((a+b) == (c-d)) && ((e*f) > (g/h)) || ((i+j) < (k*l)));
print(((x/y) == (z - p)) && ((q + r) < (s * t)) || ((u - v) > (w / aa)));
"""

    # Edge-ish arithmetic (zeros and ones; still no unary)
    base12 = """
print((a + 0) * (b - 0) / (c + 1));
print(((a + b) / (c + 1)) > ((d * e) - (f + 0)));
print((a + (b * (c + (d * (e + (f * (g + (h * (i + (j * (k + l))))))))))) == m);
"""

    # Alternating operators with nested groups
    base13 = """
print((a+(b*(c+(d*(e+(f*(g+(h*(i+j))))))))) > (k + (l*(m + (n*o)))));
print(((a*b) + (c/d) && (e+f) > (g*h)) || ((i/j) < (k-l)));
"""
    # Logical associativity torture (left-to-right check)
    base14 = """
print((a > b) || (b > c) && (c == d) || (d < e) && (e > f) || (f == g));
"""

    # Division with grouped subexpressions (no zero literal)
    base15 = """
print((a + b) / (c + d) + (e * f) / (g + h) - (i * j) / (k + l));
print(((a*b) / (c + d)) == ((e + f) / (g*h)) && ((i + j) > (k - l)));
"""

    # Spaced comparisons around big arithmetic
    base16 = """
print(( a + b * c - d / e ) > ( f * g + h / i - j ));
print(( (a+b) * (c+d) - (e+f) / (g+h) ) == ( i*j - k/l ));
"""

    declare = """
let a=2;
let b=3;
let c=5;
let d=7;
let e=11;
let f=13;
let g=17;
let h=19;
let i=23;
let j=25;
let k=27;
let l=28;

let m=29;
let n=31;
let o=35;
let p=37;
let q=41;
let r=43;
let s=47;
let t=53;
let u=59;
let v=61;
let w=67;
let x=71;
let y=73;
let z=79;

let aa = 101;
let bb = 203;
let cc = 503;
let dd = 239;
let ee = 133;
let ff = 159;
let gg = 259;
let hh = 359;
let ii = 459;
let jj = 1001;
let kk = 1187;
"""

    a,b,c,d,e,f,g,h,i,j,k,l = 2,3,5,7,11,13,17,19,23, 25, 27, 28
    m,n,o,p,q,r,s,t,u,v,w,x,y,z = 29,31,35,37,41,43,47,53,59,61,67,71,73,79
    aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk = 101,203,503,239,133,159,259,359,459,1001,1187
    codeobj = compile_trivial(declare + base1)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        a + b * c - d / e,
        x * y + z / w - p * q + r / s,
        a * b * c * d + e / f - g * h / i,
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base2)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        a + b * c > d - e / f,
        (a + b) * c < d / (e - f),
        (a * b + c / d) == (e * f - g / h),
        u + v * w == x + y / z,
        m * n - p / q > r * s + t / u,
        (a + b) / (c + d) < (e + f) / (g + h),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base3) 
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        a + b > c and d < e,
        a * b == c or d / e > f,
        a + b * c == d and e < f * g,
        x / y < z or p * q == r - s,
        a + b > c and d + e < f and g == h,
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base4)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        ((a + b) * (c - d)) / ((e + f) * (g - h)),
        (((a + b) * (c - d)) / (e + f)) > ((g * h) - (i / j)),
        ((a*b) + (c/d) - (e*f)) == ((g/h) + (i*j) - (k/l)),
        (((a+b)+(c+d))*((e+f)-(g+h))) < ((i*j)/(k+l)),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base5)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        ((((((a + b) * (c + d)) - (e / f)) + (g * (h + i))) / (j + (k * l))) == (m + (n * (o + p)))),
        ((((a+b)-((c*d)+(e/f))) * ((g+h)/(i+j))) > (((k*l)-(m/n))+((o+p)-(q/r)))),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base6)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (a + b > c) and (d + e > f) and (g + h > i) and (j + k > l),
        (a*b == c*d) or (e/f < g/h) or (i + j > k - l) or (m*n == o/p),
        (a < b) and (b < c) and (c < d) and (d < e) and (e < f) and (f < g),
        (a == b) or (b == c) or (c == d) or (d == e) or (e == f),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base7)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        a + b > c and d + e == f or g * h < i,
        a*b + c == d and e/f < g or h - i > j,
        x + y * z < p and q == r + s or t / u > v,
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base8)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (((((a + b)))) * (((c - d)))) == (((e * f))),
        (((a+b)*(c+d)) - ((e+f)/(g+h))) > (((i+j) * (k+l)) - (m+n)),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base9)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
a   +    b*c   -   d/  e,
a+b*c
    -d/e
    +f*g,

(a+b)
and
(c<d)
or
(e==f),

x*y
/
( a + b + c + d + e )
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base10)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        a+b*c - d/e + f*g - h/i + j*k - l/m + n*o - p/q + r*s - t/u + v*w - x/y + z*aa - bb/cc + dd*ee - ff/gg + hh*ii - jj/kk
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base11)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        ((a+b) == (c-d)) and ((e*f) > (g/h)) or ((i+j) < (k*l)),
        ((x/y) == (z - p)) and ((q + r) < (s * t)) or ((u - v) > (w / aa)),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base12)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (a + 0) * (b - 0) / (c + 1),
        ((a + b) / (c + 1)) > ((d * e) - (f + 0)),
        (a + (b * (c + (d * (e + (f * (g + (h * (i + (j * (k + l))))))))))) == m,
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base13)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (a+(b*(c+(d*(e+(f*(g+(h*(i+j))))))))) > (k + (l*(m + (n*o)))),
        ((a*b) + (c/d) and (e+f) > (g*h)) or ((i/j) < (k-l)),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base14)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (a > b) or (b > c) and (c == d) or (d < e) and (e > f) or (f == g),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)



    codeobj = compile_trivial(declare + base15)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        (a + b) / (c + d) + (e * f) / (g + h) - (i * j) / (k + l),
        ((a*b) / (c + d)) == ((e + f) / (g*h)) and ((i + j) > (k - l)),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)


    codeobj = compile_trivial(declare + base16)
    code = codeobj.code
    pretty_ir(code)
    out, vm = run_vm(codeobj)
    lines = out.splitlines()
    print(lines)
    assert vm.sp == 0
    expected = [
        ( a + b * c - d / e ) > ( f * g + h / i - j ),
        ( (a+b) * (c+d) - (e+f) / (g+h) ) == ( i*j - k/l ),
    ]
    for line, expect in zip(lines, expected):
        assert parse_optional(line) == approx(expect)

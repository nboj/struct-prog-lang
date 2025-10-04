from enum import IntEnum
import random, timeit

class Op(IntEnum):
    NOP=0; ADD=1; SUB=2; MUL=3; DIV=4; PUSHK=5; LOADG=6; STOREG=7; JZ=8; JMP=9; HALT=10

# Make a dense stream
N = 1_0000
ops = [random.choice(list(Op)) for _ in range(N)]
A   = [random.randint(0, 100) for _ in range(N)]
B   = [random.randint(0, 100) for _ in range(N)]

def run_if():
    ip = 0; acc = 0
    while ip < N:
        op = ops[ip]
        if op == Op.PUSHK:
            acc += A[ip]; ip += 1
        elif op == Op.ADD:
            acc += 1; ip += 1
        elif op == Op.SUB:
            acc -= 1; ip += 1
        elif op == Op.MUL:
            acc *= 2; ip += 1
        elif op == Op.DIV:
            acc //= 2 or 1; ip += 1
        elif op == Op.JMP:
            ip = (ip + A[ip]) % N
        elif op == Op.JZ:
            ip = ip + 1 if acc else (ip + B[ip]) % N
        elif op == Op.NOP:
            ip += 1
        elif op == Op.LOADG:
            acc ^= A[ip]; ip += 1
        elif op == Op.STOREG:
            acc ^= B[ip]; ip += 1
        elif op == Op.HALT:
            break
        else:
            ip += 1
    return acc

def run_match():
    ip = 0; acc = 0
    while ip < N:
        op = ops[ip]
        match op:
            case Op.PUSHK:
                acc += A[ip]; ip += 1
            case Op.ADD:
                acc += 1; ip += 1
            case Op.SUB:
                acc -= 1; ip += 1
            case Op.MUL:
                acc *= 2; ip += 1
            case Op.DIV:
                acc //= 2 or 1; ip += 1
            case Op.JMP:
                ip = (ip + A[ip]) % N
            case Op.JZ:
                ip = ip + 1 if acc else (ip + B[ip]) % N
            case Op.NOP:
                ip += 1
            case Op.LOADG:
                acc ^= A[ip]; ip += 1
            case Op.STOREG:
                acc ^= B[ip]; ip += 1
            case Op.HALT:
                break
            case _:
                ip += 1
    return acc

# table with bound callables (cold ops); inline a few hot ones if you want
def _nop(vm,a,b): vm['ip'] += 1
def _add(vm,a,b): vm['acc'] += 1; vm['ip'] += 1
def _sub(vm,a,b): vm['acc'] -= 1; vm['ip'] += 1
def _mul(vm,a,b): vm['acc'] *= 2; vm['ip'] += 1
def _div(vm,a,b): vm['acc'] //= 2 or 1; vm['ip'] += 1
def _pushk(vm,a,b): vm['acc'] += a; vm['ip'] += 1
def _jmp(vm,a,b): vm['ip'] = (vm['ip'] + a) % N
def _jz(vm,a,b): vm['ip'] = vm['ip'] + 1 if vm['acc'] else (vm['ip'] + b) % N
def _loadg(vm,a,b): vm['acc'] ^= a; vm['ip'] += 1
def _storeg(vm,a,b): vm['acc'] ^= b; vm['ip'] += 1
def _halt(vm,a,b): vm['ip'] = N

DISPATCH = [None]*(max(o.value for o in Op)+1)
for op,fn in {
    Op.NOP:_nop, Op.ADD:_add, Op.SUB:_sub, Op.MUL:_mul, Op.DIV:_div,
    Op.PUSHK:_pushk, Op.JMP:_jmp, Op.JZ:_jz, Op.LOADG:_loadg, Op.STOREG:_storeg, Op.HALT:_halt
}.items():
    DISPATCH[op.value] = fn

def run_table():
    vm = {'ip':0,'acc':0}
    while vm['ip'] < N:
        ip = vm['ip']; op = ops[ip]
        DISPATCH[op](vm, A[ip], B[ip])
    return vm['acc']

def bench(fn, reps=10_000):
    return min(timeit.repeat(fn, number=1, repeat=5))

print("if/elif: ", bench(run_if))
print("match:   ", bench(run_match))
print("dispatch:", bench(run_table))

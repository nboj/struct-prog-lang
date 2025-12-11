
import time


start = time.perf_counter_ns()
MOD = 1000000007

def hot_loop_mod(n):
    i = 0
    a = 1
    b = 2
    c = 3
    acc = 0
    while i < n:
        acc = (acc + (a * b % MOD) + c - (a + b)) % MOD
        a = (a + 1) % MOD; b = (b + 2) % MOD; c = (c + 3) % MOD
        i = i + 1
    
    return acc


def lcg(x):
    return (1664525 * x + 1013904223) % 4294967296

def call_heavy(n):
    i = 0
    x = 0
    while i < n: 
        x = lcg(x)
        i = i + 1;
    return x;

def fib_rec(n):
    if n <= 1: return n 
    return fib_rec(n-1) + fib_rec(n-2)

print("hot_loop:", hot_loop_mod(5_000_000));
print("call_heavy:", call_heavy(5_000_000));
print("fib_rec:", fib_rec(35));

end = time.perf_counter_ns()
print(f"=== {(end-start)/1e+9} ===")

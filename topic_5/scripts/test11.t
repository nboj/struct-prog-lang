let MOD = 1000000007;

fn hot_loop_mod(n) {
    let i = 0; let a = 1; let b = 2; let c = 3; let acc = 0;
    while i < n {
        acc = (acc + (a * b % MOD) + c - (a + b)) % MOD;
        a = (a + 1) % MOD; b = (b + 2) % MOD; c = (c + 3) % MOD;
        i = i + 1;
    }
    return acc;
}

fn lcg(x) {              // cheap, bounded call target
    return (1664525 * x + 1013904223) % 4294967296;
}
fn call_heavy(n) {
    let i = 0; let x = 0;
    while i < n { x = lcg(x); i = i + 1; }
    return x;
}

fn fib_rec(n) {
    if n <= 1 { return n; }
    return fib_rec(n-1) + fib_rec(n-2);
}

print("hot_loop:", hot_loop_mod(5_000_000));
print("call_heavy:", call_heavy(5_000_000));
print("fib_rec:", fib_rec(35));

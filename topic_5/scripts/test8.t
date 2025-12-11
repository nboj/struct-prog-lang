// --- sanity & arithmetic ---
let A = 2;
let B = 3;
let C = 5;
let D = 7;
let E = 11;

print("arith:", ((A+B)*(C+D) - (E - (A*B)) + (C*C*C) / (A+B+C)) - (D/(E - B)));

print("chain-comp 1:", A < B && B < C && C < D && D < E);
print("chain-comp 2:", (A+B) > C > 0);        // left-assoc check: (A+B) > C, then (that) > 0

print("logic mix 1:", (true && false) || (false || (A < B)));
print("logic mix 2:", (A == B) || (B == C) || (C == C) || (C != C));

// --- shadowing & blocks ---
let x = 10;
let y = 20;
print("xy:", x, y);
{
    let x = 99;
    let y = x + 1;
    print("inner xy:", x, y);      // expect 99, 100
    {
        let x = y + 1;
        print("inner2 x:", x);     // expect 101
    }
    print("after inner2 x:", x);   // still 99
}
print("outer xy:", x, y);          // still 10, 20

// --- reassignment & equality ---
x = y + 5;
print("x->", x);
print("eq25:", x == 25);
print("neq24:", x != 24);

// --- while + continue + break (skip 5 & 13; stop at 17) ---
let i = 0;
while (true) {
    if (i == 5 || i == 13) {
        i = i + 1;
        continue;
    }
    print("loop i:", i);
    if (i == 17) {
        break;
    }
    i = i + 1;
}
print("after loop i");

// --- nested loops (rect grid), continue odd j, break early on sentinel ---
let I = 1;
while (I < 4 || I == 4) {
    let J = 1;
    while (J < 6 || J == 6) {
        // treat "odd" J via (J/2)*2 != J (works if division is real)
        if ( (J/2)*2 != J ) {
            J = J + 1;
            continue; // only even J print
        }
        print("cell:", I, J);
        if (I == 3 && J == 4) {
            print("break inner at 3,4");
            break;
        }
        J = J + 1;
    }
    if (I == 3) {
        print("break outer at I=3");
        break;
    }
    I = I + 1;
}
print("after nested");

// --- subtraction-based GCD (no modulo) ---
let gx = 252;
let gy = 198;
print("gcd start:", gx, gy);
while (gx != gy) {
    if (gx > gy) {
        gx = gx - gy;
    } else {
        gy = gy - gx;
    }
}
print("gcd(252,198) =", gx);

// --- Collatz walk (using parity check without %) ---
let n = 27;
let steps = 0;
print("collatz start:", n);
while (n != 1) {
    // even if (n/2)*2 == n
    if ( (n/2)*2 == n ) {
        n = n / 2;
    } else {
        n = 3*n + 1;
    }
    steps = steps + 1;
    if (steps > 200) { // guard
        print("collatz bail");
        break;
    }
}
print("collatz end:", n, "steps:", steps);

// --- sorting 5 vars by pairwise swaps (ascending) ---
let s1 = 42;
let s2 = 5;
let s3 = 77;
let s4 = 1;
let s5 = 33;

print("unsorted:", s1, s2, s3, s4, s5);

// simple bubble-ish network of compares & swaps
let changed = true;
while (changed == true) {
    changed = false;

    if (s1 > s2) { let t = s1; s1 = s2; s2 = t; changed = true; }
    if (s2 > s3) { let t = s2; s2 = s3; s3 = t; changed = true; }
    if (s3 > s4) { let t = s3; s3 = s4; s4 = t; changed = true; }
    if (s4 > s5) { let t = s4; s4 = s5; s5 = t; changed = true; }

    // sweep backwards too to stir it well
    if (s4 < s3) { let t = s4; s4 = s3; s3 = t; changed = true; }
    if (s3 < s2) { let t = s3; s3 = s2; s2 = t; changed = true; }
    if (s2 < s1) { let t = s2; s2 = s1; s1 = t; changed = true; }
}
print("sorted:", s1, s2, s3, s4, s5);

// --- boolean identities & short-circuit-ish structure ---
let P = true;
let Q = false;

print("demorgan1:", !(P && Q) == (!P || !Q));
print("demorgan2:", !(P || Q) == (!P && !Q));
print("xor-ish:", (P || Q) && !(P && Q));

// chained comparisons stress
print("chain-eq:", (1 == 1) == true);
print("chain-gt:", (10 > 5) > 0);
print("chain-lt:", (5 < 10) < 1);

// --- deep nesting & breaks ---
let depth = 0;
{
    depth = depth + 1;
    {
        depth = depth + 1;
        {
            depth = depth + 1;
            print("depth:", depth); // expect 3
        }
        print("depth mid:", depth); // 3
    }
    print("depth out:", depth); // 3
}
print("depth final:", depth);

// --- guarded division & equality sanity ---
let num = 100;
let den = 7;
let q = num / den;
let r = num - q*den; // remainder via q
print("div parts:", q, r, "recompose:", q*den + r == num);

// --- while with compound condition & continue branch ---
let t = 0;
while (t < 15 && true) {
    if (t > 3 && t < 7) {
        t = t + 1;
        continue;
    }
    if (t == 12) {
        break;
    }
    print("t:", t);
    t = t + 1;
}

print("DONE ALL");

fn fib(target) {
    if target <= 0 {
        return target;
    }
    let a = 1;
    let b = 0;
    let i = 0;
    while (i < target) {
        let new_a = a+b;
        b = a;
        a = new_a;
        i += 1;
    }
    return b;
}

let i = 0;
while i < 100 {
    let answer = fib(i);
    print(answer);
    i += 1;
}

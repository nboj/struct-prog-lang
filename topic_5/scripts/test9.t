fn fib(a) {
    if a <= 0 {
        return 0;
    } 
    if a == 1 {
        return 1;
    }
    return fib(a-1) + fib(a-2);
}

let i = 0;
while i <= 30 {
    let answer = fib(i);
    print(answer);
    i += 1;
}

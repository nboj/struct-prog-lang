let x = 10;
let y = 20;

print("sum:", x + y);

x = y + 5;
print("x:", x);

print("eq1:", x == 25);
print("eq2:", x == 24);

let z = 0;
z = x + y;
print("z:", z);

if x == 25 {
    print("branch:", "then-1");
    print("branch:", "then-2");
} else {
    print("branch:", "else");
}

print("done");

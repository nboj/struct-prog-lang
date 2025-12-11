let a = 2;
a = a + 4;

let b = a;
b = a + b + 2;

print(a + b);
print("a" + " + " + "b");


print(a);
print(b);
if (a > b) {
    print("a is greater than b");
} else {
    print("a is less than b");
}


if (b < a) {
    print("b is less than a");
} else {
    print("b is greater than a");
}


if (a == b) {
    print("a == b");
}

if (a != b) {
    print("a != b");
}

if (true == false) {
    print("true == false????");
} else {    
    print("true != false");
}


if (true > false) {
    print("true > false");
} else {
    print ("false > true");
}

if (b > a > 0) {
    print("a, b:", a, b);
    print("a is between b and 0");
} else {
    print("a is NOT between b and 0");
}

if (b + 5*10 == 29) {
    print("b + 5*10 = 29 == 29");
} else {
    print("Error");
}

if (b==29) {
    print("ERROR b should not be 29");
} else {
    print("b is not 29");
}

if (a > 2 == 1) {
    print("a is > than 2 and a>2 = true so: true == 1");
} else {
    print("err: a is <= than 2 and/or a>2 = fals so: false == 1");
}


if (1==1) {
    if (2==2) {
        if (a < b) {
            print("A IS LESS THAN B");
        } else {
            print("Error");
        }
        print("outside A IS LESS THAN B");
    } else {
        print("Error");
    }
    print("outside outside A IS LESS THAN B");
}

saveRamThreshold(7)
int a = 10;
int b = 20;
void test {
    if a < b {
        a = a << b;
    }
}
test();

asm(add r5, r6, r7);
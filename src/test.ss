saveRamThreshold(7)
int a = 10;
int b = 20;
void yes {
    if a < b {
        a = a << b;
    }
}
yes();

asm(add r5, r6, r7);
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : \
    ((b) < (c) ? (b) : (c)))

int levenstein(char *s1, int s1len, char *s2, int s2len) {
    unsigned int x, y, lastdiag, olddiag;
    unsigned int column[s1len+1];
    for (y = 1; y <= s1len; y++)
        column[y] = y;
    for (x = 1; x <= s2len; x++) {
        column[0] = x;
        for (y = 1, lastdiag = x-1; y <= s1len; y++) {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y-1] + 1,
                lastdiag + (s1[y-1] == s2[x-1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return(column[s1len]);
}

#include <stdio.h>
#include <stdlib.h>

#include "string_util.h"

#define COUNT 10000
char uu[2*COUNT][1000];
int uulen[2*COUNT];

int main() {
 for (int i=0; i < 2*COUNT; i++) {
  int len = (rand() % 998) + 2;
  uulen[i]= len;
  for(int j = 0; j < len; j++) {
   uu[i][j] = (char) (rand() % 256);
  }
 }
 printf("Processing\n");
 for (int i=0; i < COUNT; i++) {
  levenstein(uu[2*i], uulen[2*i], uu[2*i+1], uulen[2*i+1]);
 }
 return 0;
}



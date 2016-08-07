#ifndef __RANDOM64_H__
#define __RANDOM64_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

uint64_t rand64(void);
double randdbl(void);
void seed64(uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif

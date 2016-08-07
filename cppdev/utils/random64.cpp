/*
 * Based on xorshift128plus algorithm from Wikipedia.
 * http://en.wikipedia.org/wiki/Xorshift
 */

#include <sys/time.h>

#include "random64.h"
#include "dev_utils.h"

#define UINT64_C(val) (val##ULL)

/* xorshift64star */

/* The state must be seeded so that it is not everywhere zero. */
static uint64_t x;
 
static uint64_t rand_xs64s(void) {
	x ^= x >> 12; // a
	x ^= x << 25; // b
	x ^= x >> 27; // c
	return x * UINT64_C(2685821657736338717);
}

static void seed_xs64s(uint64_t seed) {
	x = seed;
} 

/* xorshift1024star */

/* The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed,  we suggest to seed a xorshift64* generator and use its
   output to fill s. */
 
static uint64_t s[16];
static int p;
 
static uint64_t rand_xs1024s(void) {
	uint64_t s0 = s[ p ];
	uint64_t s1 = s[ p = (p+1) & 15 ];
	s1 ^= s1 << 31; // a
	s1 ^= s1 >> 11; // b
	s0 ^= s0 >> 30; // c
	return ( s[p] = s0 ^ s1 ) * UINT64_C(1181783497276652981);
}

static void seed_xs1024s(uint64_t seed) {
    seed_xs64s(seed);
    for (int i = 0; i < sizeof(s) / sizeof(s[0]); i++) {
		s[i] = rand_xs64s();
	}
}

uint64_t rand64(void) {
	return(rand_xs1024s());
}

void seed64(uint64_t seed) {
	seed_xs1024s(seed);
}

/* distributed equally in [0,1[ */
double randdbl(void) {
	return ((rand64() & ((1ULL << 63) - 1)) / (double) (1ULL << 63));
}

/* Default initialization is automatic */

static void init_seed64(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	seed64(tv.tv_sec * 1000000ULL + tv.tv_usec);
}

StaticCall(init_seed64);

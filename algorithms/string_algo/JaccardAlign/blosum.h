/*
 * blosum.h
 *
 *  Created on: Feb 26, 2016
 *      Author: morel
 */

#ifndef BLOSUM_H_
#define BLOSUM_H_

#define BLOSUM90

#ifdef BLOSUM62
#include "blosum62.h"
#elif defined(BLOSUM90)
#include "blosum90.h"
#endif

#endif /* BLOSUM_H_ */

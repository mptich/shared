/*
 * GenerateSuffix.cpp
 *
 *  Created on: Feb 21, 2016
 *      Author: morel
 *
 *  Generates a dictionary of suffix -> similar suffixes, based on the affinity table
 *  (BLOSUM table for Proteins). Each suffix is of class JacSuffix.
 *  Input parameters:
 *  <suffix size>
 *  <score size> (how many bits the score will take)
 *  <number of amino acids to use> (22, or 23 (including ANY (X)))
 *
 */

//TODO: UNFINISHED

#include <stdint.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <functional>

using namespace std;

#include "blosum.h"

#include "JaccardAlignCmds.h"

typedef struct AminDist_ {
	int amin_;
	int dist_;
} AminDist;

static void calculateClosestAmins(AminDist *am, int aminCount) {
	for (int i=0; i<aminCount; i++) {
	}
}

int GenerateSuffix(int argc, char **argv) {
	if (argc < 2) {
		cout << "Missing args <input file name> <suffix length>" << endl;
		return -1;
	}

	int suffixSize;
	int scoreSize;
	int aminCount;
	istringstream(argv[0]) >> suffixSize;
	istringstream(argv[1]) >> scoreSize;
	istringstream(argv[2]) >> aminCount;

	// Output map
	map<uint32_t, set<uint32_t> > expandDict;

	// Mapping of each aminoacid to closest aminoacids (based on BLOSUM table).
	// Vector is sorted
	AminDist amDist[aminCount * aminCount];
	calculateClosestAmins(amDist, aminCount);



	return 0;
}




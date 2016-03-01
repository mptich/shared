/*
 * JaccardUtils.cpp
 *
 *  Created on: Feb 23, 2016
 *      Author: morel
 */

#include <stdint.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <assert.h>

using namespace std;

#include "blosum.h"

#include "JaccardUtils.h"


set<string> jaccardSetFromString(string const &s, int len, int *pRepeats) {
	set<string> ret;
	int repeats = 0;
	for(int i=0; i <= s.length()-len; i++) {
		string suff = s.substr(i,len);
		if (pRepeats)
			if (ret.find(suff) != ret.end())
				repeats++;
		ret.insert(suff);
	}
	if (pRepeats)
		*pRepeats = repeats;
	return ret;
}

// Builds a dictionary <Suffix> -> <Its position in the protein>. Suffixes with multiple
// positions are discarded.
map<string, int> jaccardDictFromString(string const &s, int len, int *pRepeats) {
	map<string, int> ret;
	set<string> repeatSet;
	for(int i=0; i <= s.length()-len; i++) {
		string suff = s.substr(i,len);
		if (ret.find(suff) == ret.end())
			ret[suff] = i;
		else
			repeatSet.insert(suff);
	}
	for (set<string>::const_iterator it = repeatSet.begin(); it != repeatSet.end(); it++)
		ret.erase(*it);
	if (pRepeats)
		*pRepeats = repeatSet.size();
	return ret;
}

string const *jaccardFindBestMatch(string const &s, set<string> const &setStr) {
	int score = INT_MIN;
	string const *bestMatch = NULL;
	for (set<string>::const_iterator it = setStr.begin(); it != setStr.end(); it++) {
		int newScore = jaccardSuffixScore(s, *it);
		if (newScore > score) {
			score = newScore;
			bestMatch = &(*it);
		}
	}
	return bestMatch;
}

int jaccardSuffixScore(const string &s1, const string &s2) {
	assert(s1.length() == s2.length());
	int score = 0;
	for (string::const_iterator it1 = s1.begin(), it2 = s2.begin();
			it1 != s1.end(); it1++, it2++)
		score += blosumMatrix[blosumCharToOrdinal[(int) *it1]]
							  [blosumCharToOrdinal[(int) *it2]];
	return score;
}





/*
 * JaccardUtils.h
 *
 *  Created on: Feb 23, 2016
 *      Author: morel
 */

#ifndef JACCARDUTILS_H_
#define JACCARDUTILS_H_

set<string> jaccardSetFromString(string const &s, int len, int *pRepeats);
string const *jaccardFindBestMatch(string const &s, set<string> const &setStr);
int jaccardSuffixScore(const string &s1, const string &s2);

#endif /* JACCARDUTILS_H_ */

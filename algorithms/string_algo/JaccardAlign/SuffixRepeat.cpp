/*
 * SuffixRepeat.cpp
 *
 *  Created on: Feb 21, 2016
 *      Author: morel
 *
 *	Reads a file where each line represents a string, and determines how many of
 *	suffixes of each string are encountered in more than one place inside the string.
 *	Input parameters:
 *	<input file name>
 *	<suffix length>
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <boost/algorithm/string.hpp>

using namespace std;

#include "JaccardAlignCmds.h"
#include "JaccardUtils.h"

int SuffixRepeat(int argc, char **argv) {
	if (argc < 2) {
		cout << "Missing args <input file name> <suffix length>" << endl;
		return -1;
	}
	ifstream infile(argv[0]);
	if (!infile.is_open()) {
		cout << "File " << argv[0] << " could not be opened" << endl;
		return -1;
	}
	int len;
	istringstream(argv[1]) >> len;
	string line;
	int totalRep = 0, totalLen = 0;
	while (getline(infile, line)) {
		int repeats;
		boost::algorithm::trim_right(line); // It is actually not needed
		set<string> substrSet = jaccardSetFromString(line, len, &repeats);
		cout << endl << "length " << line.length() << " repeats " << repeats << endl;
		totalRep += repeats;
		totalLen += line.length();
	}
	cout << "Totals " << totalLen << "/" << totalRep << " " <<
			((double) totalRep) / totalLen << endl;
 	return 0;
}


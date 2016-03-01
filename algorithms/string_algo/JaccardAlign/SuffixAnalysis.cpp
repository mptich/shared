/*
 * SuffixAnalysis.cpp
 *
 *  Created on: Feb 23, 2016
 *      Author: morel
 *
 *  Quickly generates output file for analysis in Python, in the format:
 *  file1 <protein1 line>	<protein1 length>
 *  file2 <protein2 line> <protein2 length>
 *  1
 *  suffix1	<suffix1 pos> bestSuffix2 <bestSuffix2 pos>
 *  suffix1	<suffix1 pos> bestSuffix2 <bestSuffix2 pos>
 *  ...
 *  2
 *  suffix1	<suffix1 pos> bestSuffix2 <bestSuffix2 pos>
 *  ...
 *  file1 <protein1 line>	<protein1 length>
 *  file2 <protein2 line> <protein2 length>
 *  ...
 *  end
 *
 *  Input parameters:
 *  1st file with proteins
 *  2nd file with proteins
 *  Output file name
 *  Suffix length
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

using namespace std;

#include "blosum.h"

#include "JaccardAlignCmds.h"
#include "JaccardUtils.h"


int SuffixAnalysis(int argc, char **argv) {
	if (argc < 4) {
		cout << "Missing <input file 1> <input file 2> <output file> "
				"<suffix length>" << endl;
		return -1;
	}
	ifstream infile1(argv[0]);
	ofstream outfile(argv[2]);
	int suffixLen;
	istringstream(argv[3]) >> suffixLen;
	if (!infile1.is_open()) {
		cout << "File " << argv[0] << " could not be opened" << endl;
		return -1;
	}

	string line1, line2;
	int linenum1=0;
	while (getline(infile1, line1)) {
		linenum1++;
		cout << "Processing line " << linenum1 << endl;
		ifstream infile2(argv[1]);
		if (!infile2.is_open()) {
			cout << "File " << argv[1] << " could not be opened" << endl;
			return -1;
		}
		int linenum2=0;
		while (getline(infile2, line2)) {
			linenum2++;

			map<string, int> dict1 = jaccardDictFromString(line1, suffixLen, NULL);
			map<string, int> dict2 = jaccardDictFromString(line2, suffixLen, NULL);
			set<string> set1;
			set<string> set2;
			for (map<string, int>::const_iterator it = dict1.begin();
					it != dict1.end(); it++) {
				set1.insert(it->first);
			}
			for (map<string, int>::const_iterator it = dict2.begin();
					it != dict2.end(); it++) {
				set2.insert(it->first);
			}

			outfile << "start" << endl;
			outfile << "file1\t" << linenum1 << '\t' << line1.length() << endl;
			outfile << "file2\t" << linenum2 << '\t' << line2.length() << endl;

			outfile << "1" << endl;
			for (set<string>::const_iterator it = set1.begin(); it != set1.end(); it++) {
				string const *bestMatch = jaccardFindBestMatch(*it, set2);
				outfile << *it << '\t' << dict1[*it] << '\t' <<
						*bestMatch << '\t' << dict2[*bestMatch] << endl;
			}

			outfile << "2" << endl;
			for (set<string>::const_iterator it = set2.begin(); it != set2.end(); it++) {
				string const *bestMatch = jaccardFindBestMatch(*it, set1);
				outfile << *it << '\t' << dict2[*it] << '\t' <<
						*bestMatch << '\t' << dict1[*bestMatch] << endl;
			}

			outfile << "end" << endl;
		}
	}
	return 0;
}





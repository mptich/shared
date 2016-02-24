//============================================================================
// Name        : JaccarAlign.cpp
// Author      : Misha Orel
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <string>
#include <iostream>
#include <map>
#include <boost/assign.hpp>

using namespace std;

#include "JaccardAlignCmds.h"

typedef int (* JaccardAlignCommand) (int, char **);

static map<string, JaccardAlignCommand> cmdOptMap =
	boost::assign::map_list_of
	("suffRpt", SuffixRepeat)
	("genSuff", GenerateSuffix)
	("suffAnalysis", SuffixAnalysis);

int main(int argc, char ** argv) {
	if (argc < 2) {
		cout << "Missing command argument" << endl;
		return -1;
	}
	JaccardAlignCommand func = cmdOptMap[argv[1]];
	if (func)
		return (*func)(argc-2, argv+2);
	else {
		cout << "Wrong command argument" << endl;
		return -1;
	}
}

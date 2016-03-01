/*
 * JacSuffix.h
 *
 *  Created on: Feb 21, 2016
 *      Author: morel
 *
 *  Defines class JacSuffix.
 */

#ifndef JACSUFFIX_H_
#define JACSUFFIX_H_

#include <stdint.h>
#include <string>
#include <assert.h>

#include "blosum.h"

using namespace std;

class JacSuffix {
public:

	JacSuffix(string s, int score) {
		assert(s.length() == len_);
		assert((1 << scoreSize_) > score);
		uint32_t rep = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s[i];
			rep *= aminCount_;
			rep += blosumCharToOrdinal[(int) c];
		}
		rep_ = (rep << scoreSize_) | score;
	}

	JacSuffix(uint32_t rep) : rep_(rep) {}

	uint32_t rep() { return rep_; }

	string content(int *pscore) {
		if (pscore)
			*pscore = rep_ & ((1 << scoreSize_) - 1);
		uint32_t rep = rep_ >> scoreSize_;
		char cs[len_+1];
		cs[len_] = '\0';
		for (int i = len_-1; i>=0; i--) {
			cs[i] = blosumCoveredAmins[rep % len_];
			rep /= len_;
		}
		string s;
		s.assign(cs);
		return s;
	}

	static void aminCount(size_t aminCount) { aminCount_ = aminCount; }
	static size_t aminCount() { return aminCount_; }
	static void len(size_t len) { len_ = len; }
	static size_t len() { return len_; }
	static void scoreSize(size_t scoreSize) { scoreSize_ = scoreSize; }
	static size_t scoreSize() { return scoreSize_; }

private:
	static size_t len_;
	static size_t aminCount_;
	static int scoreSize_;
	uint32_t rep_;
};


#endif /* JACSUFFIX_H_ */

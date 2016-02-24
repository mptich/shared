# Converts text BLOSUM matrix into C++ header
# Original taken from GitHub:
# https://gist.github.com/jwintersinger/1870047
# Then modified.
# Usage: python blosum.py blosum62.txt blosum.h

import sys
import os

from string_algo.blosum_matrix import BlosumMatrix

protein_string = "ARNDCQEGHILKMFPSTWYVBZX*"

file_template1 = \
"""
// This file is autogenerated. Do not change it.

#ifndef __BLOSUM_MATRIX_H__
#define __BLOSUM_MATRIX_H__

static const char *blosumCoveredAmins = "ARNDCQEGHILKMFPSTWYVBZX*";

static unsigned char blosumCharToOrdinal[] = {
"""

file_template2 = \
"""
};
static int blosumMatrix[][24] = {
"""

file_template3 = \
"""
};

#endif
"""

def main():
  if len(sys.argv) != 3:
    sys.exit('Usage: python %s matrix_filename C++_headdr_file_name' % \
      sys.argv[0])
  matrix_filename = sys.argv[1]
  header_filename = sys.argv[2]
  matrix = BlosumMatrix(matrix_filename)

  with open(header_filename, 'w') as f:
    f.write(file_template1)

    output = ""
    for i in range(256):
      c = chr(i)
      if c in protein_string:
        val = protein_string.index(c)
      else:
        val = 0xff
      if output:
        output += ", "
      output += ("%d" % val)

    f.write(output)
    f.write(file_template2)

    output = ""
    for c1 in protein_string:
      for c2 in protein_string:
        if output:
          output += ", "
        output += ("%s" % str(matrix.lookup_score(c1, c2)))

    f.write(output)
    f.write(file_template3)

if __name__ == '__main__':
  main()

# Converts text BLOSUM matrix into C++ header
# Original taken from GitHub:
# https://gist.github.com/jwintersinger/1870047
# Then modified.
# Usage: python blosum.py blosum62.txt blosum.h

import sys

protein_string = "ARNDCQEGHILKMFPSTWYVBZX"

file_template1 = \
""" // This file is autogenerated. Do not change it.

static unsigned char charToOrdinal[] = {
"""

file_template2 = \
"""
};
static int blosumMatrix[][23] = {
"""

class InvalidPairException(Exception):
  pass

class Matrix:
  def __init__(self, matrix_filename):
    self._load_matrix(matrix_filename)

  def _load_matrix(self, matrix_filename):
    with open(matrix_filename) as matrix_file:
      matrix = matrix_file.read()
    lines = matrix.strip().split('\n')

    header = lines.pop(0)
    columns = header.split()
    matrix = {}

    for row in lines:
      entries = row.split()
      row_name = entries.pop(0)
      matrix[row_name] = {}

      if len(entries) != len(columns):
        raise Exception('Improper entry number in row')
      for column_name in columns:
        matrix[row_name][column_name] = entries.pop(0)

    self._matrix = matrix

  def lookup_score(self, a, b):
    a = a.upper()
    b = b.upper()

    if a not in self._matrix or b not in self._matrix[a]:
      raise InvalidPairException('[%s, %s]' % (a, b))
    return self._matrix[a][b]


def main():
  if len(sys.argv) != 3:
    sys.exit('Usage: python %s matrix_filename C++_headdr_file_name' % \
      sys.argv[0])
  matrix_filename = sys.argv[1]
  header_filename = sys.argv[2]
  matrix = Matrix(matrix_filename)

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
        output += ("%s" % matrix.lookup_score(c1, c2))

    f.write(output + "\n};\n")


if __name__ == '__main__':
  main()
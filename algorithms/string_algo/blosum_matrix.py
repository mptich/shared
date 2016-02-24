import sys

class BlosumMatrix:
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
        matrix[row_name][column_name] = int(entries.pop(0))

    self._matrix = matrix

  def lookup_score(self, a, b):
    a = a.upper()
    b = b.upper()

    if a not in self._matrix or b not in self._matrix[a]:
      print('Could not find pair [%s, %s] in the matrix' % (a, b))
      sys.exit(-1)
    return self._matrix[a][b]

  def aminoSet(self):
    return "".join(self._matrix.keys())


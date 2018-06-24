import glob
import shutil
import random
import sys
import os

def _usage():
    print('USAGE: random_file_choice.py <source dir> <dest dir> <count> [move]')
    sys.exit(0)

try:
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    count = int(sys.argv[3])
    move = False
    if (len(sys.argv) == 5) and (sys.argv[4] == 'move'):
        move = True
except:
    _usage()
    
file_list = list(glob.glob(src_dir + '/*'))
random.shuffle(file_list)
file_list = file_list[:count]

for fn in file_list:
    _, fn_base = os.path.split(fn)
    fn_out = dest_dir + '/' + fn_base
    if move:
        shutil.move(fn, fn_out)
    else:
        shutil.copy(fn, fn_out)

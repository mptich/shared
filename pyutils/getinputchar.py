# Getting input character without having to press return
# From StackOverflow:
# http://stackoverflow.com/questions/510357/ \
# python-read-a-single-character-from-the-user

import sys
import tty
import termios
import os
import signal

class _Getch:
    def __init__(self):
        pass

    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if ord(ch) == 3: # Ctrl-C
            os.kill(0, signal.SIGINT)
        return ch

getInputChar = _Getch()

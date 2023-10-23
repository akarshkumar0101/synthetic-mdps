
# get the current host name
import socket
import os
import sys


def main():
    print('hello')
    print(socket.gethostname())
    print(sys.executable)


if __name__ == '__main__':
    main()



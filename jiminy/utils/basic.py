import os

def create_directory(dirname):
    dirs = dirname.split('/')
    root = dirs[0]
    for directory in dirs:
        if not os.path.exists(root):
            os.mkdir(root)
        if root != directory:
            root += directory

def get_lines_ff(fname):
    with open(fname, mode='r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

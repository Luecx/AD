# replace all files in the given folder
import os
import shutil
import sys
import pathlib
import re
import argparse


def replace_keep_case(word, replacement, text):
    def func(match):
        g = match.group()
        if g.islower(): return replacement.lower()
        if g.istitle(): return replacement.title()
        if g.isupper(): return replacement.upper()
        return replacement

    return re.sub(word, func, text, flags=re.I)


def process_file(directory, file, pattern, replacement):
    out_name = replace_keep_case(pattern, replacement, file)
    print(file, out_name)

    with open(directory / file) as infile:
        with open(directory / out_name, 'w') as outfile:
            for line in infile:
                line = replace_keep_case(pattern, replacement, line)
                outfile.write(line)
    os.remove(f"{directory / file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Automatic creation for kernel and host code for the AD project. '
                    'It assumes that each function is split into a function manager, the host and kernel code.')

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('--dir', type=str, help='root directory of where all the operations are stored',
                        default='../../src/operations/')
    optional.add_argument('--base', type=str, help='base function which will be copied',
                        default='template_func')
    required.add_argument('--func', type=str, help='new function name which will be created', required=True)

    args=parser.parse_args()

    path = pathlib.Path(args.dir).resolve()
    path_folder_base = (path / args.base).resolve()
    path_folder_func = (path / args.func).resolve()

    print(path_folder_base)
    print(path_folder_func)

    # copy files first
    shutil.copytree(path_folder_base, path_folder_func)

    # rename all files and their content

    # get all files using glob
    files = os.listdir(path_folder_func)
    print(files)

    for f in files:
        full_path = path_folder_func / f
        if os.path.isfile(full_path):
            process_file(path_folder_func, f, args.base, args.func)

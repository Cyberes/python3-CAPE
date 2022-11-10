import argparse
import glob
import os
from pathlib import Path

import CAPE as cape

parser = argparse.ArgumentParser(description='Automated extraction of comic book panels.')
parser.add_argument('input', type=str, help='Path to file or folder you want to extract the panels from.')
parser.add_argument('output', type=str, help='Path to folder you want to extract the panels to. If it doesn\'t exist it will be created.')
parser.add_argument('-d', '--directory-structure', action='store_true', help='Create the directory structure of the input.')
args = parser.parse_args()

input_path = Path(args.input)  # handle stuff like ~
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
allowed_files = ['.jpg', '.jpeg', '.png']


def process(file):
    global args, input_path, output_dir, allowed_files
    if os.path.splitext(os.path.basename(file))[-1] in allowed_files:
        out_path = output_dir
        if args.directory_structure:
            out_path = os.path.join(output_dir, Path(file).parent.absolute().name)
            if Path(out_path).absolute().name != Path(input_path).absolute().name:
                os.makedirs(out_path, exist_ok=True)
            else:
                out_path = output_dir
        print(f'{file} -> {out_path}')
        cape.crop_page(file, out_path)


if os.path.isdir(input_path):
    for file in glob.glob(f'{input_path}/*'):
        if os.path.isdir(file):
            for file_r in glob.glob(f'{file}/*'):
                process(file_r)
        else:
            process(file)

elif os.path.isfile(input_path):
    if os.path.splitext(os.path.basename(input_path))[-1] in allowed_files:
        print(f'{input_path} -> {output_dir}')
        cape.crop_page(input_path, output_dir)

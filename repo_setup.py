import sys
import os
import fileinput

root_dir = os.getcwd()
old_name = 'pytemplate'
module_name = root_dir.split('/')[-1]


print(f' Detected module name {module_name}. Is this correct (y/n) ?')
yes_no = input()

if yes_no == 'y':
    os.system('git config core.hooksPath .hooks')
    os.system(f'git remote set-url origin https://github.com/edmundsj/{module_name}.git')

    files_to_search = ['setup.py', '.hooks/pre-commit',
    '.github/workflows/python-package-conda.yml']

    for filename in files_to_search:
        with fileinput.FileInput(
                filename, inplace=True, backup='.bak') as fh:
            for line in fh:
                print(line.replace(old_name, module_name), end='')


    os.system('git push')

#!/bin/bash

git config filter.jupyter_clean.clean \
    "`git rev-parse --show-toplevel`/_venv/bin/jupyter nbconvert \
    --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

if [[ `uname` = Linux ]] ; then
    . $ANACONDA_HOME/etc/profile.d/conda.sh
else
    eval $('/c/Program Files/Anaconda3/Scripts/conda.exe' 'shell_bash', 'hook')
fi

python_version=`head -1 requirements.txt`

conda env create -p _venv --file environment.yml

conda activate ./_venv

if [[ `uname` != Linux ]] ; then
    conda install pywin32
fi

conda deactivate

shopt -q login_shell && read -n1 -r -p "Press any key to continue..." key

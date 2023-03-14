#!/bin/bash


if [[ `uname` = Linux ]] ; then
    JUPYTER=`git rev-parse --show-toplevel`/.venv/bin/jupyter
    . $ANACONDA_HOME/etc/profile.d/conda.sh
else
    JUPYTER=`git rev-parse --show-toplevel`/.venv/Scripts/jupyter
    eval "$('/c/Program Files/Anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
fi

git config filter.jupyter_clean.clean \
    "$JUPYTER nbconvert \
    --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"


conda env create -p .venv --file environment.yml

conda activate ./.venv

if [[ `uname` != Linux ]] ; then
    conda install pywin32
fi

NO_INSTALL_REQUIREMENTS=true python setup.py develop

conda deactivate

shopt -q login_shell && read -n1 -r -p "Press any key to continue..." key

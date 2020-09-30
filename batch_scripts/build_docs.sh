#!/bin/bash

if [[ `uname` = Linux ]] ; then
    . ~/miniconda3/etc/profile.d/conda.sh
else
    eval "`sed '1 s|^.*$|_CONDA_EXE="/c/Program Files/Anaconda3/Scripts/conda.exe"|;s/\$_CONDA_EXE/"$_CONDA_EXE"/g' /c/Program\ Files/Anaconda3/etc/profile.d/conda.sh`"
fi

conda activate ./_venv

conda install sphinx
conda install -c conda-forge m2r2 nbsphinx ipykernel make --yes
pip install nbsphinx-link
pip install sphinx-rtd-theme

jupyter_path = "./_venv/Scripts/jupyter"

cd doc
make html

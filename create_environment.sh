#!/bin/bash

git config filter.jupyter_clean.clean \
    "`git rev-parse --show-toplevel`/_venv/bin/jupyter nbconvert \
    --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

if [[ `uname` = Linux ]] ; then
    . ~/miniconda3/etc/profile.d/conda.sh
else
    eval "`sed '1 s|^.*$|_CONDA_EXE="/c/Program Files/Anaconda3/Scripts/conda.exe"|;s/\$_CONDA_EXE/"$_CONDA_EXE"/g' /c/Program\ Files/Anaconda3/etc/profile.d/conda.sh`"
fi

python_version=`head -1 requirements.txt`

conda create -p _venv \"$python_version\" pip --yes
conda activate ./_venv

sed '1d' requirements.txt | \
while read req ; do
    conda install $req --yes || pip install $req
done

conda deactivate

shopt -q login_shell && read -n1 -r -p "Press any key to continue..." key

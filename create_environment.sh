#!/bin/bash

git config filter.jupyter_clean.clean \
    "`git rev-parse --show-toplevel`/_venv/bin/jupyter nbconvert \
    --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

exit 0

if [[ `uname` = Linux ]] ; then
    . ~/miniconda3/etc/profile.d/conda.sh
else
    eval "`sed '1 s|^.*$|_CONDA_EXE="/c/Program Files/Anaconda3/Scripts/conda.exe"|;s/\$_CONDA_EXE/"$_CONDA_EXE"/g' /c/Program\ Files/Anaconda3/etc/profile.d/conda.sh`"
fi

python_version=`head -1 requirements.txt`

echo $python_version

conda create -p _venv --file ./requirements_CONDA.txt
echo "Environment created"

conda activate ./_venv
echo "Environment activated"

cat requirements_PIP.txt | \
while read req ; do
	echo $req
    pip install $req
done

echo "Pip packages installed"

conda deactivate
echo "Environment deactivated"

shopt -q login_shell && read -n1 -r -p "Press any key to continue..." key

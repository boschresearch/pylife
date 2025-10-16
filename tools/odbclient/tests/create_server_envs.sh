#!/bin/bash


if [ "$(uname)" == "Linux" ]; then
    CONDABASE=`conda info --base`
    CONDA_SH=${CONDABASE}"/etc/profile.d/conda.sh"
    source $CONDA_SH
else
    eval "$('/c/Program Files/Anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
fi

conda create -n odbserver-2022 python=2.7 --yes
conda activate odbserver-2022
pip install -e ../odbserver

conda create -n odbserver-2023 python=2.7 --yes
conda activate odbserver-2023
pip install -e ../odbserver

conda create -n odbserver-2024 python=3.10 --yes
conda activate odbserver-2024
pip install -e ../odbserver

conda create -n odbserver-2022-version-mismatch python=2.7 --yes
conda activate odbserver-2022-version-mismatch
pip install pylife-odbserver

conda create -n odbserver-2023-version-mismatch python=2.7 --yes
conda activate odbserver-2023-version-mismatch
pip install pylife-odbserver

conda create -n odbserver-2024-version-mismatch python=3.10 --yes
conda activate odbserver-2024-version-mismatch
pip install pylife-odbserver==2.2.0a6

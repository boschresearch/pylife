call conda create -y -n _venv2 --file ./requirements_CONDA.txt

call conda activate _venv2

for /F "tokens=*" %%A in (requirements_PIP.txt) do pip install %%A

call pytest

call conda deactivate
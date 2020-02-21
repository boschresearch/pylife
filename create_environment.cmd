call conda create -y -n _venv --file ./requirements_CONDA.txt

call conda activate _venv

for /F "tokens=*" %%A in (requirements_PIP.txt) do pip install %%A
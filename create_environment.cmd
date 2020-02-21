call conda create -y -n _pylife --file ./requirements_CONDA.txt

call conda activate _pylife

for /F "tokens=*" %%A in (requirements_PIP.txt) do pip install %%A
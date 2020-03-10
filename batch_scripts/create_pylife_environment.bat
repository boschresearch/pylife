call conda env remove -p ./_venv

call conda create -y -p _venv --file requirements_CONDA.txt

call conda activate ./_venv

call pip install -r requirements_PIP.txt
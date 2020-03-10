call conda env remove --name _pylife

call conda create -y -n _pylife --file requirements_CONDA.txt

call conda activate _pylife

call pip install -r requirements_PIP.txt
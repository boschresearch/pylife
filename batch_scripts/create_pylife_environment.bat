call conda env remove -p ./_venv

call conda env create -p _venv --file environment.yml

call conda activate ./_venv

call conda install pywin32

set NO_INSTALL_REQUIREMENTS=true

call python setup.py develop
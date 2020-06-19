conda activate ./_venv

conda install sphinx=2.4.4
conda install -c conda-forge m2r nbsphinx ipykernel make --yes
pip install nbsphinx-link 
pip install sphinx-rtd-theme

jupyter_path = "./_venv/Scripts/jupyter"

cd doc
make html

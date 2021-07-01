call conda env remove -p ./_venv

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%/_venv/Scripts/jupyter

call git config filter.jupyter_clean.clean "%jupyter_path% nbconvert --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

call conda env create -p _venv --file environment.yml

call conda activate ./_venv

call conda install pywin32

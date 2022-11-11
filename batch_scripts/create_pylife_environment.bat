call conda env remove -p ./.venv

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%/.venv/Scripts/jupyter

call git config filter.jupyter_clean.clean "%jupyter_path% nbconvert --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

call conda create -p .venv pip=20.2 pandoc setuptools_scm "python==3.9"

call conda activate ./.venv

call pip install -e .[testing,docs,analysis,pymc,extras,tsfresh]

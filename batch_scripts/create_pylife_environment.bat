call conda env remove -p ./.venv2

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%/.venv2/Scripts/jupyter

call git config filter.jupyter_clean.clean "%jupyter_path% nbconvert --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

rem call conda env create -p .venv2 --file environment.yml
call conda create -p .venv2 pip=20.2 python=3.7

call conda activate ./.venv2

rem call conda install pywin32
call pip install -e .[testing,docs]
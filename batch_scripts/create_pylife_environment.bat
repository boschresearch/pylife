:: go one level up, if started from within the batch_scripts directory
set BATCH_PATH=%cd%

:: test if current path contains "batch_scripts"
if not x%BATCH_PATH%==x%BATCH_PATH:batch_scripts=% (
	echo "cd"
    cd ..
)

call conda deactivate

call conda env remove -p ./.venv

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%/.venv/Scripts/jupyter

rem filter disabled because it takes very long on windows and makes sourcetree basically unusable
rem call git config filter.jupyter_clean.clean "%jupyter_path% nbconvert --stdin --stdout --to notebook --ClearOutputPreprocessor.enabled=True"

call conda create -p .venv --yes pip=20.2 pandoc setuptools_scm "python==3.9"

call conda activate ./.venv

call pip install -e .[testing,docs,analysis,pymc,extras,tsfresh]

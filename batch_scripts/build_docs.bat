:: go one level up, if started from within the batch_scripts directory
set BATCH_PATH=%cd%

:: test if current path contains "batch_scripts"
if not x%BATCH_PATH%==x%BATCH_PATH:batch_scripts=% (
	echo "cd"
    cd ..
)

call conda activate ./.venv

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%.venv/Scripts/jupyter

call sphinx-build -j1 -b html docs/ docs/_build

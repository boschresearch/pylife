:: go one level up, if started from within the batch_scripts directory
set BATCH_PATH=%cd%

:: test if current path contains "batch_scripts"
if not x%BATCH_PATH%==x%BATCH_PATH:batch_scripts=% (
	echo "cd"
    cd ..
)

call conda activate ./.venv

if exist flake8.log del flake8.log
call flake8 --exit-zero --exclude=.venv > flake8.log || exit /B 1
:: go one level up, if started from within the batch_scripts directory
set BATCH_PATH=%cd%

:: test if current path contains "batch_scripts"
if not x%BATCH_PATH%==x%BATCH_PATH:batch_scripts=% (
	echo "cd"
    cd ..
)

call conda activate ./.venv

call python batch_scripts\test_cov_analysis.py E:\pyLife_coverage\ || exit /B 1
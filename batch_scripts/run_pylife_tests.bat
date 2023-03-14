:: go one level up, if started from within the batch_scripts directory
set BATCH_PATH=%cd%

:: test if current path contains "batch_scripts"
if not x%BATCH_PATH%==x%BATCH_PATH:batch_scripts=% (
	echo "cd"
    cd ..
)


call conda activate ./.venv

call python setup.py build_ext --inplace --force --define CYTHON_TRACE

call python -m pytest -v -ra --cache-clear --junit-xml=junit.xml --cov-report xml:coverage_report.xml --cov-report html:coverage_report --cov=pylife || exit /B 1

call pytest -mdemos
call conda activate _pylife

call python -m pytest -v -ra --cache-clear --junit-xml=junit.xml --cov-report xml:coverage_report.xml --cov-report html:coverage_report --cov=../pylife/ || exit /B 1

if exist flake8.log del flake8.log
call flake8 --exit-zero --exclude=virtual-env > flake8.log || exit /B 1

call conda deactivate


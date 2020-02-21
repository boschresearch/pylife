call conda create -y -n _venv --file ./requirements_CONDA.txt

call conda activate _venv

for /F "tokens=*" %%A in (requirements_PIP.txt) do pip install %%A

call python -m pytest -v -ra --cache-clear --junit-xml=junit.xml --cov-report xml:coverage_report.xml --cov-report html:coverage_report --cov=pylife/ || exit /B 1

call conda deactivate
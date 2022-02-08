call conda activate ./.venv

rem call pip install -r docs/requirements.txt

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%.venv/Scripts/jupyter

call sphinx-build -b html docs/ docs/_build

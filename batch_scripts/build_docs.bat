call conda activate ./.venv

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%.venv/Scripts/jupyter

call sphinx-build -j1 -b html docs/ docs/_build

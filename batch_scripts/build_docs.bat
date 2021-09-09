call conda activate ./_venv

call pip install -r docs/requirements.txt

for /f "tokens=*" %%a in ('git rev-parse --show-toplevel') do (set repo_path=%%a)

set jupyter_path=%repo_path%_venv/Scripts/jupyter

call sphinx-build -b html docs/ docs/_build

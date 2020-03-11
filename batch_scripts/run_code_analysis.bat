call conda activate ./_venv

if exist flake8.log del flake8.log
call flake8 --exit-zero --exclude=_venv > flake8.log || exit /B 1
call conda activate ./.venv

if exist flake8.log del flake8.log
call flake8 --exit-zero --exclude=.venv > flake8.log || exit /B 1
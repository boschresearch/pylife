rem go one level up
cd ..

call conda activate ./.venv

rem make sure that jupyter is installed
call pip install jupyter

call jupyter notebook --notebook-dir ./demos


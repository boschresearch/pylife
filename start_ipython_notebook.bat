@echo off
setlocal
pushd %~dp0

set VENVTOOLSDIR=venv-tools

call %VENVTOOLSDIR%\setup_venv || exit /b 1
call %VENVTOOLSDIR%\activate_venv || exit /b 1

set IPYTHON_NB_ROOT_DIR=%~dp0
if exist "%IPYTHON_NB_ROOT_DIR%\IPython Notebooks" (
    echo The directory name for "IPython Notebooks" should be changed
    echo to "IPythonNotebooks" (without whitespace^).
    echo Please rename your directory!
    pause
    exit /b 1
)
set "IPYTHON_NB_DIR=%IPYTHON_NB_ROOT_DIR%\demos"
if not exist "%IPYTHON_NB_DIR%" mkdir "%IPYTHON_NB_DIR%"
jupyter notebook "--notebook-dir=%IPYTHON_NB_DIR%"

endlocal
:: vim: et ts=4 sw=4

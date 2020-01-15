@echo off

set "BASEDIR=%CD%"
if not "%1"=="" set "BASEDIR=%1"

if exist "%BASEDIR%\.condarc" set "CONDARC=%BASEDIR%\.condarc"

:: <= requires nothing
call %~dp0retrieve_python_version.cmd
:: => defines PYTHON_VERSION and MAJOR_PYTHON_VERSION

set ENV_NAME=_venv%PYTHON_VERSION%
set ENV_DIR=%ENV_NAME%

setlocal
for %%D in ("%BASEDIR%\dummy.txt") do set BASEDIR=%%~dpD
for %%D in ("%BASEDIR%\dummy.txt") do set BASEDIR83=%%~dpD

echo Activating environment %BASEDIR%%ENV_DIR%
if "%BASEDIR83%" == "%BASEDIR%" goto skiptransformpathto8dot3
	set BASEDIR=%BASEDIR83%
:skiptransformpathto8dot3
endlocal & set BASEDIR=%BASEDIR%

:check_for_activate_command
where activate 1>NUL 2>NUL
if errorlevel 1 (
    where useanaconda.bat 1>NUL 2>NUL
    if errorlevel 1 (
        echo Cannot find 'conda'. Is Anaconda installed? Is it on the PATH?
        timeout /t 30
        exit /b 1
    )
    echo Calling 'useanaconda.bat' as conda doesn't seem to be on the PATH
    call useanaconda.bat
    goto check_for_activate_command
)

call activate "%BASEDIR%%ENV_DIR%" || exit /b 1

popd
exit /b 0
:: vim: et ts=4 sw=4

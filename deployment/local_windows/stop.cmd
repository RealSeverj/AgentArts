@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if errorlevel 1 goto failed
echo Only the owned local inference process is managed. The cloud server is unchanged.
pause
exit /b 0
:failed
echo Stop was refused or failed. No unrelated process is managed.
pause
exit /b 1

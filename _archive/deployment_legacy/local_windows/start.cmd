@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -WaitSeconds 0
if errorlevel 1 goto failed
echo Launch requested. Use status.cmd to check model readiness.
pause
exit /b 0
:failed
echo Local inference did not start. See the error above.
pause
exit /b 1

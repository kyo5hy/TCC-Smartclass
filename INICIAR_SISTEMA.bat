@echo off
title SmartClass - Painel de Controle
color 0A

:MENU
cls
echo ===================================================
echo      SMARTCLASS - GESTAO PEDAGOGICA (SAAS)
echo ===================================================
echo.
echo 1 - INICIAR SISTEMA (Abre Servidores e Pagina Web)
echo 2 - PARAR SISTEMA (Desliga os Servidores)
echo 3 - SAIR
echo.
set /p opcao="Escolha uma opcao (1, 2 ou 3): "

if "%opcao%"=="1" goto INICIAR
if "%opcao%"=="2" goto PARAR
if "%opcao%"=="3" goto SAIR

goto MENU

:INICIAR
echo.
echo [1/3] Iniciando o Backend API (Node.js)...
start "SmartClass_Node" cmd /c "cd node_api && node server.js"

echo [2/3] Iniciando o Motor de IA (Python)...
start "SmartClass_Python" cmd /c "cd python_ia && call venv\Scripts\activate && python main.py"

echo [3/3] Abrindo o Painel do Professor no Navegador...
:: Aguarda 3 segundos para os servidores subirem antes de abrir a tela
timeout /t 3 >nul
start "" "%CD%\frontend\index.html"

echo.
echo Servidores rodando perfeitamente em segundo plano!
pause
goto MENU

:PARAR
echo.
echo Desligando os servidores do SmartClass...
:: Procura as janelas com o nome SmartClass e força o encerramento limpo
taskkill /FI "WINDOWTITLE eq SmartClass_Node*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq SmartClass_Python*" /T /F >nul 2>&1
echo.
echo Servidores desligados com sucesso. Memoria liberada!
pause
goto MENU

:SAIR
exit
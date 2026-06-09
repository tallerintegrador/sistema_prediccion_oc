<#
    run.ps1 — Orquestador de desarrollo (Windows / PowerShell).

    Uso (desde la raíz del proyecto):
        ./scripts/run.ps1 <comando>

    Comandos:
        setup     Instala dependencias de Python (venv) y de Node (web).
        ml-a      Ejecuta el pipeline del Módulo A (pronóstico).
        ml-b      Ejecuta el pipeline del Módulo B (anomalías).
        metrics   Regenera los JSON de métricas (ml/metrics_export).
        load-db   Carga los artefactos a la base de datos (ETL).
        api       Levanta el API (uvicorn) en http://127.0.0.1:8000.
        web       Levanta el frontend (Vite) en http://localhost:5173.
        pipeline  ml-a + ml-b + metrics + load-db (todo el ML de punta a punta).
        all       pipeline + recordatorio para abrir api y web.
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('setup', 'ml-a', 'ml-b', 'metrics', 'load-db', 'api', 'web', 'pipeline', 'all')]
    [string]$Comando
)

# 'Continue' evita que el logging a stderr de Python (INFO) se interprete como
# error fatal del script; cada comando reporta su propio código de salida.
$ErrorActionPreference = 'Continue'
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

# Python del venv si existe; si no, el del PATH.
$Py = if (Test-Path "$Raiz/venv/Scripts/python.exe") { "$Raiz/venv/Scripts/python.exe" } else { "python" }

switch ($Comando) {
    'setup' {
        & $Py -m pip install -r requirements.txt
        Push-Location "$Raiz/web"; npm install; Pop-Location
    }
    'ml-a'     { & $Py -m ml.run_a }
    'ml-b'     { & $Py -m ml.run_b }
    'metrics'  { & $Py -m ml.metrics_export }
    'load-db'  { & $Py -m ml.load_to_db }
    'api'      { & $Py -m uvicorn api.app.main:app --reload --port 8000 }
    'web'      { Push-Location "$Raiz/web"; npm run dev; Pop-Location }
    'pipeline' {
        & $Py -m ml.run_a
        & $Py -m ml.run_b
        & $Py -m ml.metrics_export
        & $Py -m ml.load_to_db
    }
    'all' {
        & $Py -m ml.run_a; & $Py -m ml.run_b; & $Py -m ml.metrics_export; & $Py -m ml.load_to_db
        Write-Host "`nML listo. Ahora, en dos terminales:" -ForegroundColor Green
        Write-Host "  ./scripts/run.ps1 api"
        Write-Host "  ./scripts/run.ps1 web"
    }
}

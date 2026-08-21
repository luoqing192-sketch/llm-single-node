$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$HarnessHome = Join-Path $ProjectRoot '.dsh-home'
$SettingsTarget = Join-Path $HarnessHome 'settings.yaml'

New-Item -ItemType Directory -Force -Path $HarnessHome | Out-Null
if (-not (Test-Path $SettingsTarget)) {
    Copy-Item (Join-Path $ProjectRoot 'harness\settings.yaml') $SettingsTarget
}

$env:DSH_HOME = $HarnessHome
$env:LOCAL_LLM_API_KEY = 'local'
# Keep the preview CLI aligned with its rc.8 plugin packages. npm's `latest`
# currently points to rc.7 while its caret dependencies resolve to rc.8.
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    npm install --global 'pnpm@10'
}
pnpm dlx `
    --allow-build='@deepseek-ai/dsh-subprocess-local' `
    --allow-build='@google/genai' `
    --allow-build='koffi' `
    --allow-build='node-pty' `
    --allow-build='protobufjs' `
    '@deepseek-ai/dsh@next' web --no-open

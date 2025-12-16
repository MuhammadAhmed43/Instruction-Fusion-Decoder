$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = $PSScriptRoot
$FlistPath = Join-Path $ProjectRoot "flist"
$OutputDir = Join-Path $ProjectRoot "temp"
$OutputFile = Join-Path $OutputDir "microprocessor.output"

# Check for Icarus Verilog
if (-not (Get-Command "iverilog" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Icarus Verilog (iverilog) is not found in your PATH." -ForegroundColor Red
    Write-Host "Please install it from: https://bleyer.org/icarus/"
    Write-Host "During installation, make sure to check 'Add executable to PATH'."
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Read and process flist
Write-Host "Reading file list..."
$FileContent = Get-Content $FlistPath
$SourceFiles = @()

foreach ($Line in $FileContent) {
    if (-not [string]::IsNullOrWhiteSpace($Line)) {
        # Replace ${CORE_ROOT} with the actual path
        # Note: The flist uses forward slashes, Windows handles them fine usually, 
        # but we'll ensure paths are correct.
        $ResolvedPath = $Line.Replace('${CORE_ROOT}', $ProjectRoot)
        $SourceFiles += $ResolvedPath
    }
}

# Compile
Write-Host "Compiling..."
$CompileCmd = "iverilog -o `"$OutputFile`" " + ($SourceFiles -join " ")
Invoke-Expression $CompileCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful." -ForegroundColor Green
    
    # Run Simulation
    Write-Host "Running simulation..."
    vvp $OutputFile
} else {
    Write-Host "Compilation failed." -ForegroundColor Red
    exit 1
}

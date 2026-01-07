
# convert.ps1
# Requires: WSL + poppler-utils installed inside WSL (pdftotext available)

$root      = $PSScriptRoot
$papersDir = Join-Path $root "data\papers"
$outputDir = Join-Path $root "data\new"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function To-WslPath([string]$winPath) {
    $p = $winPath -replace '\\','/'
    if ($p -match '^([A-Za-z]):/(.*)$') {
        $drive = $matches[1].ToLower()
        $rest  = $matches[2]
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert to WSL path: $winPath"
}

$pdfs  = Get-ChildItem -Path $papersDir -Recurse -File -Filter "*.pdf"
$total = $pdfs.Count
$counter = 0

foreach ($pdf in $pdfs) {
    $counter++
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($pdf.Name)
    $outPath  = Join-Path $outputDir ($baseName + ".txt")

    $wslPdf = To-WslPath $pdf.FullName
    $wslOut = To-WslPath $outPath

    # Call Linux pdftotext in WSL
    $cmd = "pdftotext -enc UTF-8 `"$wslPdf`" `"$wslOut`""
    wsl bash -lc $cmd | Out-Null

    Write-Progress -Activity "Converting PDFs (WSL Poppler)" -Status "Processed $counter of $total" -PercentComplete (($counter / $total) * 100)
}

Write-Host "Done. Output folder: $outputDir"
Write-Host "Text files written: " (Get-ChildItem -Path $outputDir -Filter *.txt -File | Measure-Object).Count

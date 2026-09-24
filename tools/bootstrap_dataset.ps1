<#
Prepare the VINEPICs YOLO dataset on Windows when Python is not installed.
This mirrors the curated default session split in src/config.py and keeps the
original data untouched. Generated image entries are hard links when possible.
#>
[CmdletBinding()]
param(
    [string]$AnnotationFile = "VINEPICs/data/annotations/VINEPICs_annotations.json",
    [string]$ImageRoot = "VINEPICs/data/images",
    [string]$OutputRoot = "data/yolo_dataset",
    [switch]$ForceLabels
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$annotationPath = (Resolve-Path (Join-Path $projectRoot $AnnotationFile)).Path
$imageRootPath = (Resolve-Path (Join-Path $projectRoot $ImageRoot)).Path
$outputPath = Join-Path $projectRoot $OutputRoot
$coco = Get-Content -LiteralPath $annotationPath -Raw | ConvertFrom-Json
$invariant = [Globalization.CultureInfo]::InvariantCulture

$sessionSplit = @{
    "2021-07-27" = "train"; "2021-08-23" = "val"; "2021-09-06" = "train"
    "2022-08-23-12-23-31" = "train"; "2022-08-23-12-32-48" = "train"
    "2022-08-23-14-17-26" = "train"; "2022-08-23-14-32-03" = "test"
    "2022-08-23-15-22-59" = "train"; "2022-08-23-15-32-40" = "val"
    "2022-08-23-17-02-58" = "train"; "2022-08-23-17-46-30" = "train"
    "2022-08-23-17-53-22" = "test"; "2022-09-15-12-13-19" = "train"
    "2022-09-15-12-20-49" = "train"; "2022-09-15-13-08-12" = "test"
    "2022-09-15-13-14-36" = "train"; "2022-09-15-14-02-31" = "train"
    "2022-09-15-14-08-23" = "val"
}

$annotationsByImage = @{}
foreach ($annotation in $coco.annotations) {
    $key = [string]$annotation.image_id
    if (-not $annotationsByImage.ContainsKey($key)) {
        $annotationsByImage[$key] = [Collections.Generic.List[object]]::new()
    }
    $annotationsByImage[$key].Add($annotation)
}

$splitImages = @{ train = 0; val = 0; test = 0 }
$splitAnnotations = @{ train = 0; val = 0; test = 0 }
$hardLinks = 0
$copies = 0
$existingImages = 0
$emptyLabels = 0
$invalidBoxes = 0

foreach ($image in $coco.images) {
    $normalizedName = ([string]$image.file_name).Replace("\", "/")
    $parts = $normalizedName.Split("/")
    $session = $parts[0]
    if (-not $sessionSplit.ContainsKey($session)) {
        throw "No split assignment for capture session: $session"
    }
    $split = $sessionSplit[$session]
    $fileName = $parts[-1]
    $flatName = "${session}_${fileName}"
    $source = Join-Path $imageRootPath ($normalizedName.Replace("/", [IO.Path]::DirectorySeparatorChar))
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing source image: $source"
    }
    $imageDir = Join-Path $outputPath "images/$split"
    $labelDir = Join-Path $outputPath "labels/$split"
    New-Item -ItemType Directory -Force -Path $imageDir, $labelDir | Out-Null
    $target = Join-Path $imageDir $flatName
    if (-not (Test-Path -LiteralPath $target)) {
        try {
            New-Item -ItemType HardLink -Path $target -Target $source | Out-Null
            $hardLinks++
        } catch {
            Copy-Item -LiteralPath $source -Destination $target
            $copies++
        }
    } else { $existingImages++ }

    $labelPath = Join-Path $labelDir (([IO.Path]::GetFileNameWithoutExtension($flatName)) + ".txt")
    if ($ForceLabels -or -not (Test-Path -LiteralPath $labelPath)) {
        $lines = [Collections.Generic.List[string]]::new()
        $key = [string]$image.id
        $imageAnnotations = if ($annotationsByImage.ContainsKey($key)) { $annotationsByImage[$key] } else { @() }
        foreach ($annotation in $imageAnnotations) {
            $x = [double]$annotation.bbox[0]; $y = [double]$annotation.bbox[1]
            $width = [double]$annotation.bbox[2]; $height = [double]$annotation.bbox[3]
            $x1 = [Math]::Max(0.0, [Math]::Min([double]$image.width, $x))
            $y1 = [Math]::Max(0.0, [Math]::Min([double]$image.height, $y))
            $x2 = [Math]::Max(0.0, [Math]::Min([double]$image.width, $x + $width))
            $y2 = [Math]::Max(0.0, [Math]::Min([double]$image.height, $y + $height))
            if ($x2 -le $x1 -or $y2 -le $y1) { $invalidBoxes++; continue }
            $xc = (($x1 + $x2) / 2.0) / [double]$image.width
            $yc = (($y1 + $y2) / 2.0) / [double]$image.height
            $wn = ($x2 - $x1) / [double]$image.width
            $hn = ($y2 - $y1) / [double]$image.height
            $values = @($xc, $yc, $wn, $hn) | ForEach-Object { $_.ToString("0.000000", $invariant) }
            $lines.Add("0 " + ($values -join " "))
        }
        [IO.File]::WriteAllText($labelPath, [string]::Join([Environment]::NewLine, $lines))
    }
    $annotationCount = if ($annotationsByImage.ContainsKey([string]$image.id)) { $annotationsByImage[[string]$image.id].Count } else { 0 }
    if ($annotationCount -eq 0) { $emptyLabels++ }
    $splitImages[$split]++
    $splitAnnotations[$split] += $annotationCount
}

$yamlPath = Join-Path $outputPath "dataset.yaml"
$yaml = "# Generated from VINEPICs COCO annotations`npath: .`ntrain: images/train`nval: images/val`ntest: images/test`n`nnames:`n  0: grape_cluster`n"
[IO.File]::WriteAllText($yamlPath, $yaml)
$summary = [ordered]@{
    source_annotations = $AnnotationFile.Replace("\", "/")
    seed = 42
    split_strategy = "capture-session (prevents adjacent-frame leakage)"
    session_assignments = $sessionSplit
    split_image_counts = $splitImages
    split_annotation_counts = $splitAnnotations
    stats = [ordered]@{ images = $coco.images.Count; annotations = $coco.annotations.Count; empty_images = $emptyLabels; invalid_boxes = $invalidBoxes }
    materialization = [ordered]@{ hardlinks_created = $hardLinks; copies_created = $copies; existing_images = $existingImages }
    classes = @{ "0" = "grape_cluster" }
}
$summaryJson = $summary | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $outputPath "split_summary.json"), $summaryJson)
$summaryJson

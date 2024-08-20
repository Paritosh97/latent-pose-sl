# Define environment variables
$env:BLENDER_USER_CONFIG = "./blender_config/"
$env:BLENDER_USER_SCRIPTS = "./blender_scripts/"

# Create the Blender User Config directory if it doesn't exist
if (-not (Test-Path $env:BLENDER_USER_CONFIG)) {
    New-Item -ItemType Directory -Path $env:BLENDER_USER_CONFIG | Out-Null
}

# Set the root directory to the directory where the script is located
$root_directory = Split-Path -Parent $MyInvocation.MyCommand.Path

# Function to ask for Blender path
function AskForBlenderPath {
    $blender_path = Read-Host "Enter the path of Blender"
    Set-Content -Path "$env:BLENDER_USER_CONFIG\config.txt" -Value "blender_path=$blender_path"
}

# Check if blender_path variable exists in config.txt
if (Test-Path "$env:BLENDER_USER_CONFIG\config.txt") {
    $blender_path = (Get-Content "$env:BLENDER_USER_CONFIG\config.txt") -replace "blender_path=", ""
    if (-not $blender_path) {
        AskForBlenderPath
        $blender_path = (Get-Content "$env:BLENDER_USER_CONFIG\config.txt") -replace "blender_path=", ""
    }
} else {
    AskForBlenderPath
    $blender_path = (Get-Content "$env:BLENDER_USER_CONFIG\config.txt") -replace "blender_path=", ""
}

# Find all .blend files in the root directory and subdirectories, excluding userpref.blend
$blend_files = Get-ChildItem -Path $root_directory -Recurse -Filter "*.blend" | Where-Object { $_.Name -ne "userpref.blend" }

# If there are no .blend files, display error and terminate
if ($blend_files.Count -eq 0) {
    Write-Host "Error: No .blend files found in the directory $root_directory, or only userpref.blend found"
    exit 1
}

# Get the scene number from the command-line argument, if provided
$scene_number = $args[0]

# Selection process for .blend files
if (-not $scene_number) {
    if ($blend_files.Count -eq 1) {
        $blender_file = $blend_files[0].FullName
    } else {
        Write-Host "Select a .blend file:"
        for ($i = 0; $i -lt $blend_files.Count; $i++) {
            Write-Host "$i. $($blend_files[$i].FullName)"
        }
        $scene_number = Read-Host "Enter the number of the file"
        $blender_file = $blend_files[$scene_number].FullName
    }
} else {
    $blender_file = $blend_files[$scene_number].FullName
}

# Try to run Blender with the selected file, ask for the path again if it fails
while ($true) {
    & $blender_path $blender_file
    if ($?) {
        Write-Host "Blender ran successfully."
        break
    } else {
        Write-Host "Error: Blender failed to run. Please check the path and try again."
        AskForBlenderPath
        $blender_path = (Get-Content "$env:BLENDER_USER_CONFIG\config.txt") -replace "blender_path=", ""
    }
}
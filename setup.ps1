# Create and activate Conda environment
conda create -n FastSAM python=3.10 -y
conda activate FastSAM

# Install requirements
pip install -r requirements.txt

# Create weights directory and navigate to it
New-Item -ItemType Directory -Force -Path weights
Set-Location weights

# Download the file
$url = "https://drive.usercontent.google.com/download?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv&export=download&authuser=0"
$output = "FastSAM-x.pt"
Invoke-WebRequest -Uri $url -OutFile $output

# Return to the original directory
Set-Location ..
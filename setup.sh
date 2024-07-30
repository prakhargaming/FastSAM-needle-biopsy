#!/bin/bash

conda create -n FastSAM python=3.10
conda activate FastSAM
pip install -r requirements.txt
mkdir weights
cd weights
wget "https://drive.usercontent.google.com/download?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv&export=download&authuser=0" -O FastSAM-x.pt
# Collision Risk Calculation

This repository contains code for collision risk calculation based on the NuScenes dataset.

## Instructions

### 1. Download and Set Up NuScenes Devkit
Download the [NuScenes Devkit](https://github.com/nutonomy/nuscenes-devkit) and follow the instructions to set it up.
Make sure to add the devkit to your Python path.

### 2. Clone our repo and setup the environment
```bash
cd path_to_your_projects
conda create --name risk_calc python=3.10
conda activate risk_calc
git clone https://github.com/DimSpathoulas/Collision_Risk_Calculation.git
cd Collision_Risk_Calculation
pip install -r requirements.txt
```

### 3. Download the Dataset
You should download the NuScenes dataset from the official [Website](https://www.nuscenes.org/).


### 4. Get Detections
You should run a 3d detector on your downloaded dataset and save them in your dir.

### 5. The final project dir should look like this




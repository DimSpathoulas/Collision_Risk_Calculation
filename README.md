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
You can also use the mini.


### 4. Get Detections
You should run a 3d detector on your downloaded dataset and save the results in your dir.

## The final project dir should look like this
```bash
# For nuScenes Dataset         
└── Collision_Risk_Calculation
       ├── coll_risk_calc.py <-- main code
       ├── data <-- folder
              └── tracking <-- folder containing multiple results
                     ├── results_tracking.json <-- .json containing results
              └── dataset <-- folder
                     ├── maps          <-- unused
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     |── v1.0-mini     <-- metadata and annotations
```



# Collision Risk Calculation Under 3d Multi-object Motion Tracking
Part of my master thesis: **Probabilistic 3D Multi-Modal Multi-Object Tracking Using Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**


 **Note: This is a simplified published version.
 The complete module will be uploaded after thesis defense.**

 
## Overview
A collision risk module based on 3d multi-object tracks for autonomous vehicles.

## Instructions

### 1. Download and Set Up NuScenes Devkit
Download the [NuScenes Devkit](https://github.com/nutonomy/nuscenes-devkit) and follow the instructions to set it up.
Make sure to add the devkit to your Python path.

### 2. Clone our repo and setup the environment
```bash
cd path_to_your_projects
conda create --name coll_risk_calc python=3.10
conda activate coll_risk_calc
git clone https://github.com/DimSpathoulas/Collision_Risk_Calculation.git
cd Collision_Risk_Calculation
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the NuScenes dataset from the official [Website](https://www.nuscenes.org/).
You can also use the mini version.


### 4. Get Detections
Run a 3d lidar detector on your dataset and save the results.
I used [megvii](https://arxiv.org/abs/1908.09492) results from [MEGVII](https://github.com/V2AI/Det3D).
The [results](https://www.nuscenes.org/data/detection-megvii.zip) in zip.

### 5. Get Tracking Results
Run a tracking algorithm on the detections and save the results in your dir.
I used [this](https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking) repo.

## The Final Project dir
```bash
# Path_to_your_projects        
└── Collision_Risk_Calculation
       ├── coll_risk_calc.py <-- main code
       ├── tools             <-- complementary modules
       ├── data              <-- folder
              └── tracking   <-- folder containing multiple results
                     ├── results_val_tracking.json <-- .json containing results
              └── dataset    <-- folder
                     ├── maps          <-- unused
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     |── v1.0-mini     <-- metadata and annotations
```

## Run the Code - Visualize the Results
head to path_to_your_projects/Collision_Risk_Calculation and run:
```bash
python coll_risk_calc.py --data_root data\your_nuscenes_data --version your_version --tracking_file data\tracking\your_tracking_results.json --distance_thresh 12 --seconds_to_prediction 3 
```

## References and Acknowledgments

### References
- Module heavily based on the principles outlined in this paper: [Analytic_Collision_Risk_Calculation_for_Autonomous_Vehicle_Navigation](https://ieeexplore.ieee.org/document/8793264)

##Script

#!/bin/bash --login                                                             
#SBATCH -J RM2Proj                                                              
#SBATCH -p gpuA40GB                                                             
#SBATCH -G 1                                                                    
#SBATCH -n 4                                                                    
#SBATCH -t 0-01                                                                 

module purge
pip install --user scikit-learn

python RM2Proj.py

##Output
Calculated pos_weight for class imbalance: 0.7420
Training Model 1 (Baseline)...
Training Model 2 (Improved)...
  Early stopping at epoch 55 (best val AUC: 0.9289)

Model 1 AUC: 0.8417
Model 2 AUC: 0.9255
Improvement: +0.0838
Bootstrapped p-value: < 0.001

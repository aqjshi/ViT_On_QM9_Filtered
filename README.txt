Set up wandb 


for admin:
Virtual Environment Setup (expect 5 minutes and 4 GB)
pip freeze | ForEach-Object { ($_ -split '==')[0] } > requirements.txt


if windows:
python -m venv venv

./venv/scripts/activate.ps1
pip install -r requirements.txt

else:
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt


Prerequisites
pip install -r requirements.txt

Have qm9_filtered.npy in the same directory as readme


ViT
python vit_regression.py config/vit_regression_task_0.json                                  
python vit_regression.py config/vit_regression_task_1.json                                  
python vit_classification.py config/vit_classification_task_0.json                                  
python vit_classification.py config/vit_classification_task_1.json                                  

XGBoost
python xgboost_regression.py

To change the hyperaparmeters, go inside the file, ctrl optimal_config_values and update them in the code. 


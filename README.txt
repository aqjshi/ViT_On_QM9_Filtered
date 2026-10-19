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



ensure environment align with gpu 

python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU count:',torch.cuda.device_count())"

should look like this

>> 
PyTorch: 2.14.1+cu132
CUDA build: 13.2
CUDA available: True
GPU count: 1



Have qm9_filtered.npy in the same directory as readme


python eda.py config/vit_regression_whole.json    

ViT
python vit_regression.py config/vit_regression_whole.json                                  
python vit_regression.py config/vit_regression_sub.json                                  
python vit_classification.py config/vit_classification_whole.json                                  
python vit_classification.py config/vit_classification_sub.json                                  

XGBoost
python xgboost_regression.py config/xgboost_regression_whole.json                                  
python xgboost_regression.py config/xgboost_regression_sub.json                                  

To change the hyperaparmeters, go inside the config set, ctrl optimal_config_values and update them in the config file. 

python core/evaluate.py {run_hash}


see ViT run param search on

https://forge.coreweave.com/wandb/aqj-shi-blendingwaves/ViT-QM9-Regression-0?nw=nwuseraqjshi
https://forge.coreweave.com/wandb/aqj-shi-blendingwaves/ViT-QM9-Regression-1?nw=nwuseraqjshi
https://forge.coreweave.com/wandb/aqj-shi-blendingwaves/ViT-QM9-Classification-0?nw=nwuseraqjshi
https://forge.coreweave.com/wandb/aqj-shi-blendingwaves/ViT-QM9-Classification-1?nw=nwuseraqjshi


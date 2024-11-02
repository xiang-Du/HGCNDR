# Requirments
pytorch==1.12.1

scikit-learn==1.3.0

numpy==1.26.0

python==3.9.18

# How to use
if your cuda is available, you can 
run python main.py --gpu --mode separated --k 20 --layer_num 2 --order 4 --embed_dim1 64 --embed_dim2 96 --data_path ./data/Gdataset.mat --variable_weight

else 
run python main.py --mode separated --k 20 --layer_num 2 --order 4 --embed_dim1 64 --embed_dim2 96 --data_path ./data/Gdataset.mat --variable_weight
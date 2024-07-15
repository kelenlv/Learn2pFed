dataset="synthetic"
client=5
alg='Learn2pFed'
n_coefficients=1 

mkdir -p ./log/${dataset}_plot

nohup python -u demo.py  --alg $alg --n_coefficients $n_coefficients  --dataset $dataset --n_clients $client   > ./log/${dataset}_plot/${alg}_${n_coefficients}.log 2>&1 &

echo "[^_^]: finish"

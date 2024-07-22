if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SForecasting" ]; then
    mkdir ./logs/SForecasting
fi
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --patch_len 16\
      --stride 16\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'type3'\
      --ll_num 0\
      --nl_num 1\
      --use_norm 1\
      --patience 10\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/SForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SForecasting" ]; then
    mkdir ./logs/SForecasting
fi
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
      --ll_num 1\
      --nl_num 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type3'\
      --pct_start 0.4\
      --use_norm 0\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/SForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
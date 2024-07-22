if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/720LongForecasting" ]; then
    mkdir ./logs/720LongForecasting
fi
seq_len=720
model_name=Linear1

root_path_name=./data/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

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
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type3'\
      --ll_num 3\
      --nl_num 0\
      --use_norm 1\
      --itr 1 --batch_size 64 --learning_rate 0.00015 >logs/720LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
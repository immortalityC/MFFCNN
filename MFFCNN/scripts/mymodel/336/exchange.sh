if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=exchange_rate

random_seed=2021

for pred_len in 96 192
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
      --dropout 0.3 \
      --enc_in 8 \
      --weight_decay 0.004 \
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'type2'\
      --itr 1 --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done




for pred_len in  336
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
      --dropout 0 \
      --enc_in 8 \
      --weight_decay 0 \
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type1'\
      --itr 1 --batch_size 64 --learning_rate 0.000065 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
for pred_len in 720
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
      --dropout 0.1 \
      --enc_in 8 \
      --weight_decay 0 \
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 40\
      --patience 3\
      --lradj 'type1'\
      --itr 1 --batch_size 64 --learning_rate 0.000065 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
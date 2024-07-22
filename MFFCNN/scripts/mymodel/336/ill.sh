if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Linear1

root_path_name=./data/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=national_illness

random_seed=2021

for pred_len in  192
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
      --pred_len 1\
      --dropout 0.15 \
      --enc_in 7 \
      --weight_decay 0.001 \
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type3'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --pred_len 1 \
      --dropout 0.15 \
      --enc_in 7 \
      --weight_decay 0.003 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --pred_len 1 \
      --dropout 0.15 \
      --enc_in 7 \
      --weight_decay 0.002 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
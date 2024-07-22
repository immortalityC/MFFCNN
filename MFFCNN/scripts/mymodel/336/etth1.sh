if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

for pred_len in 96
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
      --dropout 0.2 \
      --enc_in 7 \
      --weight_decay 0.0035 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 192
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
      --dropout 0.2 \
      --enc_in 7 \
      --weight_decay 0.0035 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
#
#
#
#
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
      --dropout 0.2 \
      --enc_in 7 \
      --weight_decay 0.01 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.0025 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --dropout 0.15 \
      --enc_in 7 \
      --weight_decay 0.0035 \
      --patch_len 48\
      --stride 48\
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --lradj 'type1'\
      --itr 1 --batch_size 256 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
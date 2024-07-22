if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
#export CUDA_VISIBLE_DEVICES=1
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic

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
      --dropout 0.00005 \
      --weight_decay 0 \
      --enc_in 862 \
      --patch_len 48\
      --stride 48\
      --des 'Exp' \
      --train_epochs 40\
      --patience 5\
      --lradj 'type3'\
      --ll_num 0\
      --nl_num 1\
      --use_norm 1\
      --itr 1 --batch_size 8 --learning_rate  0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --dropout 0.00005 \
      --weight_decay 0 \
      --enc_in 862 \
      --patch_len 48\
      --stride 48\
      --des 'Exp' \
      --train_epochs 40\
      --patience 5\
      --lradj 'type3'\
      --ll_num 0\
      --nl_num 1\
      --use_norm 1\
      --itr 1 --batch_size 16 --learning_rate  0.0015 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
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
      --dropout 0.00005 \
      --weight_decay 0 \
      --enc_in 862 \
      --patch_len 48\
      --stride 48\
      --des 'Exp' \
      --train_epochs 40\
      --patience 5\
      --lradj 'type3'\
      --ll_num 0\
      --nl_num 1\
      --use_norm 1\
      --itr 1 --batch_size 32 --learning_rate  0.0015 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
      --dropout 0.00005 \
      --weight_decay 0 \
      --enc_in 862 \
      --patch_len 48\
      --stride 48\
      --des 'Exp' \
      --train_epochs 40\
      --patience 5\
      --lradj 'type3'\
      --ll_num 0\
      --nl_num 1\
      --use_norm 1\
      --itr 1 --batch_size 32 --learning_rate  0.0015 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
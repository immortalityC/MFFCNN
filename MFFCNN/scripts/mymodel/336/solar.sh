#export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Linear1

root_path_name=./data/
data_path_name=solar.csv
model_id_name=solar
data_name=solar

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
      --dropout 0.1 \
      --enc_in 137 \
      --weight_decay 0.001 \
      --ll_num 1\
      --nl_num 0\
      --use_norm 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'type1'\
      --itr 1 --batch_size 8 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done




# 长度增加实验变好

#for pred_len in 96 192 336 720
#do
#python -u run.py \
#        --is_training 1 \
#        --root_path ./dataset/solar/ \
#        --data_path solar_AL.csv \
#        --model_id solar_96_96_$pred_len \
#        --model $model_name \
#        --data custom \
#        --features M \
#        --seq_len 96 \
#        --pred_len $pred_len \
#        --factor 3 \
#        --enc_in 137 \
#        --dec_in 137 \
#        --c_out 137 \
#        --des 'Exp' \
#        --e_layers 3 \
#        --d_model 128 \
#        --d_ff 256 \
#        --dropout 0.1 \
#        --n_heads 4 \
#        --train_epochs 20 \
#        --patience 3 \
#        --batch_size 8 \
#        --train_epochs 10 \
#        --itr 1 \
#        --alpha -1 \
#        --learning_rate 0.001
#done



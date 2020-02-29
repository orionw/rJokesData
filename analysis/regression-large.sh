export DATA_DIR=../data
# regression task set up
export TASK_NAME=STS-B
declare -a arr=("roberta" "bert" "xlnet")
# I used batch size of 96 with xlnet-large and 128 with others
declare -a arrtwo=("roberta-large" "bert-large-uncased" "xlnet-large-cased")


for ((i=0; i<3; i++));
do
    python run_glue.py \
      --model_type "${arr[i]}"  \
      --model_name_or_path "${arrtwo[i]}" \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $DATA_DIR \
      --eval_all_checkpoints \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 128 \
      --learning_rate 2e-5 \
      --num_train_epochs 5.0 \
      --overwrite_output_dir \
      --output_dir output/large/"${arr[i]}"
done

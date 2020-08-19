mkdir output_token_128 && \
python train.py --model token --output_dir ./output_token_128/ --max_seq_length 128 --n_epochs 5 --no_finetuning --load_checkpoint

mkdir output_token_512 && \
python train.py --model token --output_dir ./output_token_512/ --max_seq_length 512 --n_epochs 5 --no_finetuning --load_checkpoint
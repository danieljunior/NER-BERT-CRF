mkdir output_crf_128_fine_last && \
python train.py --model crf --output_dir ./output_crf_128_fine_last/ --max_seq_length 128 --n_epochs 5 --bert_output last --finetuning --load_checkpoint

mkdir output_crf_128_fine_sum && \
python train.py --model crf --output_dir ./output_crf_128_fine_sum/ --max_seq_length 128 --n_epochs 5 --bert_output sum --finetuning --load_checkpoint

mkdir output_crf_512_fine_last && \
python train.py --model crf --output_dir ./output_crf_512_fine_last/ --max_seq_length 512 --n_epochs 5 --bert_output last --finetuning --load_checkpoint

mkdir output_crf_512_fine_sum && \
python train.py --model crf --output_dir ./output_crf_512_fine_sum/ --max_seq_length 512 --n_epochs 5 --bert_output sum --finetuning --load_checkpoint

mkdir output_crf_128_feat_last && \
python train.py --model crf --output_dir ./output_crf_128_feat_last/ --max_seq_length 128 --n_epochs 50 --bert_output last --no_finetuning --load_checkpoint

mkdir output_crf_128_feat_sum && \
python train.py --model crf --output_dir ./output_crf_128_feat_sum/ --max_seq_length 128 --n_epochs 50 --bert_output sum --no_finetuning --load_checkpoint

mkdir output_crf_512_feat_last && \
python train.py --model crf --output_dir ./output_crf_512_feat_last/ --max_seq_length 512 --n_epochs 50 --bert_output last --no_finetuning --load_checkpoint

mkdir output_crf_512_feat_sum && \
python train.py --model crf --output_dir ./output_crf_512_feat_sum/ --max_seq_length 512 --n_epochs 50 --bert_output sum --no_finetuning --load_checkpoint

python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone vit_b_16 \
--mode train \
--epochs 200 --batch_size 16 --lr 0.0001 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--save_dir checkpoints/phenotyping/cxr_vit_pretrain \
--fusion_type uni_cxr 

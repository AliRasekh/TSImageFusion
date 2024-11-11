python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 50 --batch_size 16 --lr 8.866e-05 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr \
--save_dir checkpoints/phenotyping/paired_ehr_pretrain \
--fusion_type uni_ehr 

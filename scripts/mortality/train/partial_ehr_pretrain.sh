python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 --lr 0.00054 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr \
--fusion_type uni_ehr --task in-hospital-mortality \
--save_dir checkpoints/mortality/partial_ehr_pretrain \
--labels_set mortality

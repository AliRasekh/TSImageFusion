python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 --lr 0.00053985 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr \
--fusion_type uni_ehr \
--save_dir checkpoints/phenotyping/partial_ehr_pretrain

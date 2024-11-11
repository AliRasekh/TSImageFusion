python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--loss uncertainty_loss \
--mode train \
--epochs 100 --batch_size 16 --lr 0.00007 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--fusion_type attention \
--save_dir checkpoints/phenotyping/attention_partial

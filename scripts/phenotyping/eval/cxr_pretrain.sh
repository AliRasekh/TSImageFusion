python main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision-backbone resnet34 \
--mode eval \
--epochs 50 --batch_size 16 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs paired_ehr_cxr \
--save_dir checkpoints/phenotyping/cxr_pretrain \
--fusion_type uni_cxr 
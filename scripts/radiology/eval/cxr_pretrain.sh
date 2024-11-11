python main.py --dim 256 \
--dropout 0.3 --mode eval \
--epochs 100 \
--vision-backbone resnet34 --data_pairs radiology \
--batch_size 16 --align 0.0 --labels_set radiology --save_dir checkpoints/cxr_rad_pretrain \
--fusion_type uni_cxr --layers 2 --vision_num_classes 14 \
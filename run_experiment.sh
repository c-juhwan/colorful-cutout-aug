DATASET_LIST=(cifar10 cifar100 tiny_imagenet)
MODEL_LIST=(resnet50 efficientnet_b0 vit_b_16)
AUG_LIST=(none cutout color_cutout_nocur color_cutout_cur mixup cutmix)
BS=32
LR=5e-5
EP=5
DEVICE=cuda:0

clear

for DATASET in ${DATASET_LIST[@]}
do

for MODEL in ${MODEL_LIST[@]}
do

python main.py --task=classification --job=preprocessing \
               --task_dataset=${DATASET} --model_type=${MODEL}

for AUG_TYPE in ${AUG_LIST[@]}
do

python main.py --task=classification --job=training \
               --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE}
python main.py --task=classification --job=testing \
               --task_dataset=${DATASET} --model_type=${MODEL} --device=${DEVICE} \
               --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} \
               --augmentation_type=${AUG_TYPE}

done # AUG_TYPE

done # MODEL

done # DATASET
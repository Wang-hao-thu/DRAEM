train_name='jier_4000_train'
#train_name='jier_ceshi'
#train_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/mvt/train.lst'
#train_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/ciwa_1/train/train.lst'
#train_list='/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD/Dataset/ciwa_2/train.lst'
train_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/jier/train/train.lst'
anomaly_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/anormaly_data/data.lst'
test_list=''
gpu_id=0
lr=0.0001
bs=8
epochs=700
data_path='../datasets/mvtec/'
#anomaly_source_path='/mnt/lustre/wanghao3/projects/DRAEM/datasets/dtd/images'
anomaly_source_path='/mnt/lustre/wanghao3/projects/DRAEM/datasets/dtd/images/'
checkpoint_path='./chechpoints'

log_path='./log_path'





srun -p mediaa --gres=gpu:1 \
        python ../train_wh.py \
        --gpu_id=$gpu_id \
        --lr=$lr \
        --train_name=$train_name \
        --train_list=$train_list \
        --anomaly_list=$anomaly_list \
        --bs=$bs \
        --epochs=$epochs \
        --data_path=$data_path \
        --anomaly_source_path=$anomaly_source_path \
        --checkpoint_path=$checkpoint_path \
        --log_path=$log_path \
        --test_list=$test_list

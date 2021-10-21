gpu_id=0
save_file='./tmp/result.txt'
test_list='/mnt/lustre/wanghao3/projects/DRAEM/data_path/mvt/test.lst'
rec_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/DRAEM_test_0.0001_700_bs8_bottle_.pckl'
seg_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/DRAEM_test_0.0001_700_bs8_bottle__seg.pckl'
test_name='ceshi'



srun -p mediaa --gres=gpu:1 \
        python ../test_wh.py \
        --gpu_id=$gpu_id \
        --test_list=$test_list \
        --save_file=$save_file \
        --rec_checkpoint=$rec_checkpoint \
        --seg_checkpoint=$seg_checkpoint \
        --test_name=$test_name

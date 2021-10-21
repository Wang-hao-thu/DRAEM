export PYTHONPATH=/mnt/lustre/wanghao3/projects/Knowlege_distillation_AD:$PYTHONPATH
gpu_id=0
test_name='ciwa_1'
#imglist='/mnt/lustre/wanghao3/projects/DRAEM/data_path/mvt/test.lst'
imglist='/mnt/lustre/wanghao3/projects/DRAEM/data_path/ciwa_1/test/test.lst'
#rec_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/DRAEM_test_0.0001_700_bs8_bottle_.pckl'
rec_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/ciwa_10.0001_700_bs8.pth'
#seg_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/DRAEM_test_0.0001_700_bs8_bottle__seg.pckl'
seg_checkpoint='/mnt/lustre/wanghao3/projects/DRAEM/tools/chechpoints/ciwa_10.0001_700_bs8_seg.pth'
out_dir=$1
partition=mediaa
split_num=16

if [ -d $out_dir ]
then
    rm -rf $out_dir
fi
mkdir $out_dir -p
num=1
in_file=$imglist

out_file_dir=${out_dir}/tmp_dir
mkdir $out_file_dir
out_file=${out_file_dir}/imglist
total_line=$(wc -l < "$in_file")
lines=$(echo $total_line/$split_num | bc -l)
line=${lines%.*}
line=$(expr $line + $num)
echo $line
split -l $line -d $in_file $out_file
#######
result_dir=$out_dir/defect
vis_dir=$out_dir/vis
mkdir $result_dir -p
for each_file in `ls ${out_file_dir}`
do
{
    sub_img_list=${out_file_dir}/${each_file}
        srun -p $partition -x SH-IDC1-10-5-30-[69,102] -n1 --gres=gpu:1 \
        python ../test_wh.py \
        --gpu_id=$gpu_id \
        --test_list=$sub_img_list \
        --save_file=$result_dir/$each_file.txt \
        --rec_checkpoint=$rec_checkpoint \
        --test_name=$test_name \
        --seg_checkpoint=$seg_checkpoint
} &
done
wait
echo done

result=$out_dir/results.all
cat $result_dir/* > $result
wc -l $result

python vis_result.py $result tmp/result_1.lst

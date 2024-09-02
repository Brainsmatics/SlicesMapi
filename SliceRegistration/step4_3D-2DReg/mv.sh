for((i=1;i<=140;i++))
do
num=`echo ${i}|awk '{printf("%04d\n",$0)}'`
m=./ann/ann_"$num".tif
f=./moving/moving_"$num".tif
res=./result/"$num"_diff.nii.gz
dst=./result_ann/"$num"_diff.nii.gz
dst2=./result_f/"$num"_diff.nii.gz
a1=./result/"$num"1Warp.nii.gz
a2=./result/"$num"0GenericAffine.mat
antsApplyTransforms -d 2 -i "$m" -o "$dst" -r "$f" -t "$a1" -t "$a2" -n NearestNeighbor
mv "$res" "$dst2"
done
for((i=1;i<=200;i++))
do
num=`echo ${i}|awk '{printf("%04d\n",$0)}'`
m=./result3/"$i"_diff.nii.gz
dst=./result3_diff/"$i"_diff.nii.gz
cp $m $dst 
done
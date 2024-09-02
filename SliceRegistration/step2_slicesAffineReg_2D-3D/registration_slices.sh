for((i=1;i<=73;i++))
do
#num=`echo ${i}|awk '{printf("%04d\n",$0)}'`
if [[ "$i" -eq "1" ]];then
f=./resize3/"$i".tif #moving
m=./resize3/"$((i+1))".tif  #fixed
else 
f=./result3/"$i"_diff.nii.gz #moving
m=./resize3/"$((i+1))".tif  #fixed
fi

out=./result3/"$((i+1))" #output file
imgs="$f,$m"

reg=${AP}antsRegistration           # path to antsRegistration

$reg -d 2 -r [ $imgs ,1] -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
            -m MI[$imgs, 1, 32] \
            -t Rigid[0.1] \
            -c [1000x800x300x0,1.e-6,50]  \
            -s 4x3x2x1vox  \
            -f 16x8x4x1 \
            --float 0  \
            -n NearestNeighbor \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]		
done
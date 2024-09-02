f=fixed_mask2_25.nii.gz
m=registration_mask2_25_diff.nii.gz
out=registration_mask3_25
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 3 -r [ $imgs ,1] -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
            -m MI[$imgs, 1 , 32] \
            -t affine[ 0.1 ] \
            -c [1000x1000x1000x0,1.e-6,20]  \
            -s 3x2x1x0vox  \
            -f 16x8x4x1\
		-m MI[$imgs, 1 ,32]  \
            -t SyN[ .1, 3, 0 ] \
            -c [1000x1000x500x100x10,1.e-8,10]  \
            -s 4x3x2x1x0vox  \
            -f 32x16x8x4x1 \
            --float 0  \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

#linear
f=fixed_mask2_25.nii.gz
m=moving_mask2_25.nii.gz
out=linear_mask2_25
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 3 -r [ $imgs ,1] -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
            -m MI[$imgs, 1 , 32] \
            -t affine[ 0.1 ] \
            -c [1000x800x100x0,1.e-6,20]  \
            -s 4x3x2x1vox  \
            -f 16x8x4x1\
            --float 0  \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

#Apply label1
antsApplyTransforms -d 3 -v 1 -i moving_mask1_25.nii.gz -o linear_out_25.nii.gz  -r fixed_mask2_25.nii.gz -t linear_mask2_250GenericAffine.mat


#nonlinear
f=fixed_mask1_25.nii.gz
m=linear_out_25.nii.gz
out=registration_mask1_25
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 3  -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
            -m MI[$imgs, 1 ,32]  \
            -t SyN[ .10, 3, 0 ] \
            -c [1000x1000x500x100x10,1.e-8,10]  \
            -s 5x4x3x2x1vox  \
            -f 32x16x8x4x1 \
            --float 0  \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

#Apply label2
antsApplyTransforms -d 3 -v 1 -i moving_mask2_25.nii.gz -o moving_out2_25.nii.gz  -r fixed_mask2_25.nii.gz -t registration_mask1_250Warp.nii.gz -t linear_mask2_250GenericAffine.mat


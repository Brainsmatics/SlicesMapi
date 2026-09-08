f=moving_mask2.tif
m=fixed_mask2.tif
out=linear_mask2
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 2  -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
			-r [$imgs,1] \
            -m MI[$imgs, 1, 32] \
			-t affine[ 0.1 ] \
            -c [1000x1000x1000x0,1.e-6,20]  \
            -s 3x2x1x0vox  \
            -f 16x8x4x1\
            --float 0  \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

#Apply label2
antsApplyTransforms -d 2 -v 1 -i fixed_mask1.tif -o fixed_out1.nii.gz  -r moving_mask2.tif -t linear_mask20GenericAffine.mat -n NearestNeighbor

#nonlinear
f=moving_mask1.tif
m=fixed_out1.nii.gz
out=registration_mask1
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 2  -v 1   \
            --use-histogram-matching true\
            --winsorize-image-intensities [0.005,0.995] \
            -m MI[$imgs, 1, 32] \
            -t SyN[ .10, 3, 0 ] \
            -c [1000x1000x500x100x10,1.e-8,10]  \
            -s 5x4x3x2x1vox  \
            -f 32x16x8x4x1 \
            --float 0  \
            -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

#Apply label2
antsApplyTransforms -d 2 -v 1 -i fixed_mask2.tif -o fixed_out2.nii.gz  -r moving_mask2.tif  -t registration_mask10Warp.nii.gz -t linear_mask20GenericAffine.mat -n NearestNeighbor
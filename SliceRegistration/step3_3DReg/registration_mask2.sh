f=fixed_mask2_25.nii.gz
m=moving_out2_25.nii.gz
out=registration_mask2_25
imgs="$f,$m"
reg=${AP}antsRegistration           # path to antsRegistration
 
$reg -d 3  -v 1  \
                    --use-histogram-matching 0 \
                    --winsorize-image-intensities [0.005,0.995] \
                    -m MI[$imgs, 1 ,32]  \
                    -t SyN[ .10, 3, 0 ] \
                    -c [1000x1000x500x10,1.e-8,10]  \
            		-s 4x3x2x1vox  \
            		-f 16x8x4x1 \
            		--float 0  \
                    -o [${out},${out}_diff.nii.gz,${out}_inv.nii.gz]

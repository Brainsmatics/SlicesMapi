#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from time import sleep
from tqdm import trange
import os
import glob
import imageio
import numpy as np
import SimpleITK as sitk
import transformations as T
import tensorflow as tf

import sys
sys.path.append('F:/ztzhang/2d_3dregistration_master/single slice/MaskSVRnet-master/')
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

###############################################################################
# Tensorflow feature wrapper

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


###############################################################################
# Dataset Locations

NIFTI_ROOT  = 'F:/ztzhang/2d_3dregistration_master/single slice/volume_data/PI/'
SAVE_DIR    = 'F:/ztzhang/2d_3dregistration_master/single slice/volume_data/traindataPI319_2/'

n_rotations = 10
n_offsets   = 100
max_z       = 528

###############################################################################
# Data Generation

SO3_GROUP   = SpecialOrthogonalGroup(3)

database    = glob.glob(NIFTI_ROOT+"*_2.tif")

for fetal_brain in database:

    print('Parsing:', fetal_brain)

    fixed_image_sitk_tmp    = sitk.ReadImage(fetal_brain, sitk.sitkFloat32)
    fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
    fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1)

    writer      = tf.python_io.TFRecordWriter(SAVE_DIR+
                                              os.path.basename(fetal_brain).replace('.tif','.tfrecord'))
    p1=np.linspace(-10,10,20)
    p2=np.linspace(-10,10,20)
    i=0
    rotations=np.zeros((400,3))
    for p11 in p1:
        for p22 in p2:
            rotations[i]=np.array([p11,p22,0])
            i=i+1

    for ii in trange(400):
        sleep(0.1)
        rotation = rotations[ii]
        a=np.pi*rotation[0]/180
        b=np.pi*rotation[1]/180
        c=0
        rotation_center = (0,0,0)
        
        rigid_euler = sitk.Euler3DTransform(rotation_center, a, b, c)


        # transform the image
        moving_image_sitk = sitk.Resample(fixed_image_sitk,rigid_euler)        
        
        npimage = sitk.GetArrayFromImage(moving_image_sitk) 

        offsets = np.random.randint(0, max_z, size=n_offsets)

        for offset in offsets:
            
            img     = npimage[offset,...]
            dz = offset
            ia = rotation[0]
            ib = rotation[1]
            
            img_raw = img.astype('float32').tostring()
            dz_raw = dz.astype('float32').tostring()
            ia_raw  = ia.astype('float32').tostring()
            ib_raw = ib.astype('float32').tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'image':    _bytes_feature(img_raw),
                'dz':      _bytes_feature(dz_raw),
                'ia':       _bytes_feature(ia_raw),
                'ib':      _bytes_feature(ib_raw)}))

            writer.write(example.SerializeToString())

    writer.close()


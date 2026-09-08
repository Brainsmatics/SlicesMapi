from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
#import imageio
import os
import sys
from PIL import Image
import imageio
import tqdm
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import generate_data.transformations as T
from configparser import ConfigParser
import shutil
import zipfile
from os.path import join,getsize
import logging
import time
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings("ignore")
from tensorflow.contrib.slim.python.slim.nets import inception
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
#选择哪一块gpu,如果是-1，就是调用cpu'
import math
import imageio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
#选择哪一块gpu,如果是-1，就是调用cpu'

conn=ConfigParser()
conn.read('./Config2.ini')

# command line argument parserA
ARGPARSER = argparse.ArgumentParser(
    description='MaskSVRnet Predict')

# directory parameters
ARGPARSER.add_argument(
        '--conn_dir',type=str,default='./Config1.ini')

ARGPARSER.add_argument(
    '--data_dir', type=str, default='./test/',
    help='The path to the dataset directory.')

ARGPARSER.add_argument(
    '--save_dir', type=str, default='./predict_result/1_',
    help='The path to the dataset directory.')

ARGPARSER.add_argument(
    '--log_dir', type=str, default='./predict_result/1_',
    help='The path to the dataset directory.')

ARGPARSER.add_argument(
    '--subject_id', type=str, default='fixed_mask2_25',
    help='Subject ID to evaluate')
ARGPARSER.add_argument(
    '--n_iter', type=int, default=100,
    help='The number of epochs to train.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")

class GetLog:
    def log(self):
        file_dir=os.getcwd()
        ltime = time.strftime('%Y%m%d',time.localtime(time.time()))
        filename=(file_dir+str(ltime)+".log")
        logger=logging.getLogger()
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = RotatingFileHandler(filename, maxBytes=1024*1024, backupCount=5, encoding="utf-8")
            formatter=logging.Formatter(fmt="%(asctime)")
            fh.setLevel(logging.ERROR)
            sh.setLevel(logging.ERROR)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        return logger



def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(0.7978845608028564 * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    attn_score = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)
    if mask is not None:
        attn_score += mask * -1e9
    attn_weight = tf.nn.softmax(attn_score)
    output = tf.matmul(attn_weight, v)
    return output, attn_weight


class MultiHeadAttention(tf.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.wq = tf.layers.Dense(embed_dim)
        self.wk = tf.layers.Dense(embed_dim)
        self.wv = tf.layers.Dense(embed_dim)
        self.dense_out = tf.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        concat_out = tf.reshape(attn_out, (batch_size, -1, self.embed_dim))
        output = self.dense_out(concat_out)
        return output


def transformer_block(x, embed_dim, num_heads, ff_dim, dropout_rate):
    attn_layer = MultiHeadAttention(embed_dim, num_heads)
    attn_out = attn_layer(x, x, x)
    x = tf.nn.dropout(x=attn_out, rate=dropout_rate)
    x = tf.contrib.layers.layer_norm(x + attn_out)

    ffn1 = tf.layers.Dense(ff_dim, activation=gelu)
    ffn2 = tf.layers.Dense(embed_dim)
    ffn_out = ffn2(ffn1(x))
    x = tf.nn.dropout(x=ffn_out, rate=dropout_rate)
    x = tf.contrib.layers.layer_norm(x + ffn_out)
    return x


def parse_tfrecord_fn(example):
    feature_desc = {
        "image": tf.FixedLenFeature([], tf.string),
        "AP1": tf.FixedLenFeature([], tf.string),
        "AP2": tf.FixedLenFeature([], tf.string),
        "AP3": tf.FixedLenFeature([], tf.string),
    }
    parsed = tf.parse_single_example(example, feature_desc)
    image = tf.decode_raw(parsed["image"], tf.float32)
    image = tf.reshape(image, RAW_SHAPE)

    ap1 = tf.decode_raw(parsed["AP1"], tf.float32)
    ap2 = tf.decode_raw(parsed["AP2"], tf.float32)
    ap3 = tf.decode_raw(parsed["AP3"], tf.float32)
    label = tf.concat([ap1, ap2, ap3], axis=0)
    return image, label


def split_image_to_patches(image):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    hh = h // 2
    ww = w // 2
    p1 = image[:hh, :ww, :]
    p2 = image[:hh, ww:, :]
    p3 = image[hh:, :ww, :]
    p4 = image[hh:, ww:, :]
    patches = [tf.image.resize_images(p, TARGET_SIZE) for p in [p1, p2, p3, p4]]
    return tf.stack(patches, axis=0)


def resnet_extract_feat(input_img, is_training):
    with tf.variable_scope("resnet_v1_50", reuse=tf.AUTO_REUSE):
        with arg_scope(resnet_v1.resnet_arg_scope()):
            net, _ = resnet_v1.resnet_v1_50(input_img, num_classes=None, is_training=is_training)
    feat = tf.squeeze(net, axis=[1, 2])
    return feat


def build_model(image_batch, is_training=True):
    image_resized = tf.image.resize_images(image_batch, TARGET_SIZE)

    # ===== coarse =====
    feat_global = resnet_extract_feat(image_resized, is_training)

    coarse_dense1 = tf.layers.dense(feat_global, 256, activation=tf.nn.relu)
    coarse_anchor = tf.nn.tanh(tf.layers.dense(coarse_dense1, OUTPUT_DIM))

    # ===== patch + transformer =====
    patch_stack = tf.map_fn(split_image_to_patches, image_batch, dtype=tf.float32)
    patch_feats = []
    for i in range(PATCH_NUM):
        patch_img = patch_stack[:, i, ...]
        patch_feat = resnet_extract_feat(patch_img, is_training)
        patch_feat = tf.layers.dense(patch_feat, EMBED_DIM)
        patch_feats.append(patch_feat)

    patch_tokens = tf.stack(patch_feats, axis=1)

    pos_emb = tf.get_variable("pos_emb", shape=[1, PATCH_NUM, EMBED_DIM], dtype=tf.float32)
    seq_tokens = patch_tokens + pos_emb

    for _ in range(NUM_LAYERS):
        seq_tokens = transformer_block(seq_tokens, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE if is_training else 0.0)

    fused_feat = seq_tokens[:, 0, :]
    residual_dense1 = tf.layers.dense(fused_feat, 256, activation=tf.nn.relu)
    residual_pred = tf.nn.tanh(tf.layers.dense(residual_dense1, OUTPUT_DIM))

    fine_anchor = coarse_anchor + residual_pred
    return coarse_anchor, fine_anchor


# ======================  pipeline（shuffle/repeat/prefetch） ======================
def build_dataset():
    tfrecord_list = glob.glob(os.path.join(DATA_DIR, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(tfrecord_list)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER)
    return dataset
def rotation_sitk(fixed_image_sitk,dz,ia,ib,ic):
    ia = np.pi*ia/180
    ib = np.pi*ib/180
    ic = np.pi*ic/180
    [xx,yy,zz] = fixed_image_sitk.GetSize()
    rotation_center = (228,180,264)
    
    rigid_euler =sitk.Euler3DTransform(rotation_center, ia, ib,ic)
    resampled_img = sitk.Resample(fixed_image_sitk,rigid_euler)
    npimage = sitk.GetArrayFromImage(resampled_img)[int(dz),...]
    return npimage
def resample_sitk_bk(fixed_image_sitk, dz, ia, ib,ic):

    # rx is the rotation of the plane, R is the rotation of the brain
    #rotationP   = np.array([0,np.pi*rotation[1]/180,np.pi*rotation[2]/180])
    rotation = np.array([np.pi*ia/180,np.pi*ib/180,np.pi*ic/180])
    R           = T.euler_matrix(*rotation)[:3,:3]
    [xx,yy,zz] = fixed_image_sitk.GetSize()
        #new_origin  = (570,450,500)
    new_origin  = (0,0,0)
        
    fixed_image_sitk.SetOrigin(new_origin)
    fixed_image_sitk.SetDirection(np.array(R.flatten()))

        # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputDirection((1,0,0,0,1,0,0,0,1)) #Identity
    resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleFilter.SetOutputSpacing([1,1,1])
    resampleFilter.SetOutputOrigin(new_origin)
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize((int(xx),int(yy),int(zz)))
    #new_origin  = (60, 60, 60) - R.dot(np.array(size) / 2) - R.dot(tx)

    #fixed_image_sitk.SetOrigin(new_origin)
    #fixed_image_sitk.SetDirection(np.array(R.flatten()))

    # resample filter
    #resampleFilter = sitk.ResampleImageFilter()
    #resampleFilter.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity
    #resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    #resampleFilter.SetOutputSpacing(spacing)
    #resampleFilter.SetOutputOrigin((0, 0, 0))
    #resampleFilter.SetDefaultPixelValue(0)
    #resampleFilter.SetSize((120, 120, 120))

    # transform the image
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)

    npimage = sitk.GetArrayFromImage(moving_image_sitk)[int(dz),...]

    return npimage

def matrix_from_anchor_points(AP1, AP2, AP3):
    v1 = AP3 - AP1
    v2 = AP2 - AP1
    n1 = np.cross(v1, v2)
    n2 = np.cross(n1, v1)

    v1_norm = v1 / np.linalg.norm(v1)  # x
    n2_norm = n2 / np.linalg.norm(n2)  # y
    n1_norm = n1 / np.linalg.norm(n1)  # z

    # SimpleITK does not like this...
    # R_recon = np.vstack((v1_norm, n2_norm, n1_norm)).T

    R_recon = np.zeros([3, 3])
    R_recon[0, 0] = v1_norm[0]
    R_recon[0, 1] = n2_norm[0]
    R_recon[0, 2] = n1_norm[0]
    R_recon[1, 0] = v1_norm[1]
    R_recon[1, 1] = n2_norm[1]
    R_recon[1, 2] = n1_norm[1]
    R_recon[2, 0] = v1_norm[2]
    R_recon[2, 1] = n2_norm[2]
    R_recon[2, 2] = n1_norm[2]

    return R_recon

def zip_file(src_dir):
    zip_name=src_dir +'.zip'
    z = zipfile.ZipFile(zip_name,'w',zipfile.ZIP_DEFLATED)
    for dirpath,dirnames,filenames in os.walk(src_dir):
        fpath=dirpath.replace(src_dir,'')
        fpath=fpath and fpath+os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath,filename),fpath+filename)
            print('==zip complete==')
    z.close()
    
def resample_sitk(fixed_image_sitk, rx, tx):

    # rx is the rotation of the plane, R is the rotation of the brain
    R = np.linalg.inv(rx)

    size        = fixed_image_sitk.GetSize()
    spacing     = fixed_image_sitk.GetSpacing()

    new_origin  = (570, 450, 700) - R.dot(np.array(size) / 2) - R.dot(tx)

    fixed_image_sitk.SetOrigin(new_origin)
    fixed_image_sitk.SetDirection(np.array(R.flatten()))

    # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity
    resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin((0, 0, 0))
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize((1140, 900, 1400))

    # transform the image
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)
    
    npimage = sitk.GetArrayFromImage(moving_image_sitk)[700,...]

    return npimage
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint32)

def _parse_function_ifind(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image' : tf.FixedLenFeature([], tf.string),
            'vec'   : tf.FixedLenFeature([], tf.string),
            'qt'    : tf.FixedLenFeature([], tf.string),
            'AP1'   : tf.FixedLenFeature([], tf.string),
            'AP2'   : tf.FixedLenFeature([], tf.string),
            'AP3'   : tf.FixedLenFeature([], tf.string)})

    image   = tf.reshape(tf.decode_raw(features['image'], tf.float32),[456,360,1]) 
    vec     = tf.reshape(tf.decode_raw(features['vec'], tf.float32),[3])
    qt      = tf.reshape(tf.decode_raw(features['qt'], tf.float32),[4])
    AP1     = tf.reshape(tf.decode_raw(features['AP1'], tf.float32),[3])
    AP2     = tf.reshape(tf.decode_raw(features['AP2'], tf.float32),[3])
    AP3     = tf.reshape(tf.decode_raw(features['AP3'], tf.float32),[3])

    return image, vec, qt, AP1, AP2, AP3

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt,R)
    I = np.identity(3,dtype=R.dtype)
    n=np.linalg.norm(I - shouldBeIdentity)
    return n<1e-6

def rotationMatrixToAngles(R):
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
    singular = sy<1e-6
    if not singular:
        x = math.atan2(R[2,1],R[2,2])
        y = math.atan2(-R[2,0],sy)
        z = math.atan2(R[1,0],R[0,0])
    else:
        x= math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],sy)
        z=0
    return np.array([x,y,z])
def main(argv):

    print('Config_dir=',argv[1])
    conn_dir=argv[1]
    conn=ConfigParser()
    try:
        conn.read(conn_dir)
    except Exception as e:
        GetLog().log().error("code:404")
    try:
        datadir=argv[2]
        #print('save_dir=',argv[3])
        savedir=argv[3]
        interval=int(argv[4])
        slice_num=int(argv[5])
    except Exception as e:
        GetLog().log().error("code:404")
    datafiles = conn.get('DIR','train_dir')
    dataset = tf.data.TFRecordDataset(datafiles)
    dataset = dataset.map(_parse_function_ifind)
    # dataset = dataset.repeat()Z
    # dataset = dataset.shuffle(FLAGS.queue_buffer)
    dataset = dataset.batch(1)
    image, vec, qt, AP1, AP2, AP3 = dataset.make_one_shot_iterator().get_next()
    

    # Nifti Volume
    try:
        fixed_path = conn.get('DIR','fixed_dir')
        fixed_image_sitk_tmp    = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
    
        ann_path = conn.get('DIR','ann_dir')
        ann_image_sitk_tmp    = sitk.ReadImage(ann_path, sitk.sitkFloat32)
        ann_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(ann_image_sitk_tmp))
    except Exception as e:
        GetLog().log().error("code:404")
    #fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1) * 255.
    # Network Definition
    image_resized = tf.image.resize_images(image, size=[299, 299])
    res1=[]
    res2=[]
    for i in range(slice_num):
        
        #imagedir=datadir+'/moving_{0:0>4d}.tif'.format(i*40)
        #savedir0=savedir+'/moving_{0:0>4d}.tif'.format(i*40)
       #savedir1=savedir+'/image_{0:0>4d}_p.tif'.format(i*40)
        #savedir2=savedir+'/atlas_{0:0>4d}_p.tif'.format(i*40)
        imagedir=datadir+'/{0:0>4d}.tif'.format(i*interval+1)
        savedir0=savedir+'/image_{0:0>4d}_t.tif'.format(i*interval+1)
        savedir1=savedir+'/image_{0:0>4d}_p.tif'.format(i*interval+1)
        savedir2=savedir+'/atlas_{0:0>4d}_p.tif'.format(i*interval+1)
        #image0    = sitk.ReadImage(imagedir, sitk.sitkFloat32)
        #fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
        #fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1)
        image0=Image.open(imagedir)
        image1=image0.resize([299,299])
    #image1 = sitk.RescaleIntensity(image1, 0, 1) * 255.
    #image1= sitk.ReadImage(datadir, sitk.sitkFloat32)
    #image1 = sitk.GetImageFromArray(sitk.GetArrayFromImage(image1))
    #image1 = sitk.RescaleIntensity(image1, 0, 1) * 255.s
    #image1 = tf.image.resize_images(image1, size=[224, 224])
        image_np=load_image_into_numpy_array(image1)
        image_np=image_np/(image_np.max()-image_np.min())*255
        image_np=np.fliplr(image_np)
        image_np_expanded= np.expand_dims(image_np,axis=0)
    # Network Definition

        coarse_pred, fine_pred = build_model(image_np_expanded, is_training=True)
        AP1_pred, AP2_pred, AP3_pred = tf.split(fine_pred, 3, axis=1)
        sess = tf.Session()
        ckpt_file = tf.train.latest_checkpoint(conn.get('DIR','model_dir'))
        tf.train.Saver().restore(sess, ckpt_file)
        #print('restoring parameters from', ckpt_file)
        _AP1,_AP2,_AP3=sess.run([AP1_pred,AP2_pred,AP3_pred],feed_dict={image_resized:image_np_expanded})
        try:
            #print(_AP1[0])
            #print(_AP2[0])
            #print(_AP3[0])

            tx = _AP2[0]*700
            #print(tx)
            rx = matrix_from_anchor_points(_AP1[0],_AP2[0],_AP3[0])
            #print(rx)
            fixed_pred=resample_sitk(fixed_image_sitk,rx,tx)
            ann_pred=resample_sitk(ann_image_sitk,rx,tx)
            ann_min=np.min(ann_pred)
            ann_max=np.max(ann_pred)
            ann_8bit=(ann_pred-ann_min)/(ann_max-ann_min+1)*255
        except Exception as e:
            GetLog().log().error("code:1001")
        #file = open(savedir+'parameter.txt','a')
        #file.write(str(Angles))
        res1.append(str(rx))
        res2.append(str(tx))
        #file.write(str(tx))
        #file.write(str(rx))
        #file.write(' ')
        #file.write(str(_ia)+' ')
        #file.write(str(_ib)+' ')
        #file.write(str(_ic)+' ')
        #file.write(str(_dz)+' ')
        #file.close()
        #fixed_pred = rotation_sitk(fixed_image_sitk, int(_dz), float(_ia),float(_ib),0)
        #ann_pred = rotation_sitk(ann_image_sitk, int(_dz), float(_ia),float(_ib),0)
        #fixed_pred = rotation_sitk(fixed_image_sitk, int(_dz*2.5), float(_ia),float(_ib),0)
        #ann_pred = rotation_sitk(ann_image_sitk, int(_dz*2.5), float(_ia),float(_ib),0)
        imageio.imsave(savedir0, np.uint8(image_np))
        imageio.imsave(savedir1, np.uint8(np.fliplr(fixed_pred)))
        #imageio.imsave(save_dir+'show.jpg', np.uint8(np.fliplr(fixed_pred)))
        #imageio.imsave(save_dir+'ann.tif', np.uint16(np.fliplr(ann_pred)))
        imageio.imsave(savedir2, np.uint8(np.fliplr(ann_8bit)))
    #imageio.imsave(save_dir+'ann.tif', np.uint16(ann_pred))
    #Angles=rotationMatrixToAngles(rx)

    #print('angle:',Angles)
    #print('tx:',tx)
    start = float(res2[3].strip(']').strip('[').split()[2]) - 3*interval if slice_num>3 else float(res2[0].strip(']').strip('[').split()[2])
    mid1 = float(res2[slice_num//2-1].strip(']').strip('[').split()[2])
    mid2 = float(res2[slice_num//2].strip(']').strip('[').split()[2])
    end = float(res2[-1].strip(']').strip('[').split()[2])

    tmp1=np.linspace(start,mid1,slice_num//2)
    tmp2=np.linspace(mid2,end,slice_num-slice_num//2)
    tmp = np.concatenate((tmp1,tmp2))

    newrs = np.zeros([len(res2),3])
    for i in range(len(res2)):

        #print(res.strip(']').strip('[').split()[2])
        newrs[i,0] = np.float(res2[i].strip(']').strip('[').split()[0])
        newrs[i,1] = np.float(res2[i].strip(']').strip('[').split()[1])
        newrs[i,2] = tmp[i]

    for i in range(len(res2)):
        rx = res1[0].strip('[').strip(']').split()
        rxx = np.zeros([3, 3])
        rxx[0, 0] = rx[0].strip(']').strip('[')
        rxx[0, 1] = rx[1].strip(']').strip('[')
        rxx[0, 2] = rx[2].strip(']').strip('[')
        rxx[1, 0] = rx[3].strip(']').strip('[')
        rxx[1, 1] = rx[4].strip(']').strip('[')
        rxx[1, 2] = rx[5].strip(']').strip('[')
        rxx[2, 0] = rx[6].strip(']').strip('[')
        rxx[2, 1] = rx[7].strip(']').strip('[')
        rxx[2, 2] = rx[8].strip(']').strip('[')
        tx = newrs[i]
        fixed_pred=resample_sitk(fixed_image_sitk,rxx,tx)
        ann_pred=resample_sitk(ann_image_sitk,rxx,tx)
        imageio.imsave(savedir+str(i*interval+1)+'_show.jpg', np.uint8(np.fliplr(fixed_pred)))
        imageio.imsave(savedir+str(i*interval+1)+'_ann.tif', np.uint8(np.fliplr(ann_pred)))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    ltime = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    logging.basicConfig(filename=os.getcwd()+"/"+str(ltime)+".log",level=logging.ERROR)
    try:
        import SimpleITK as sitk
        import tensorflow as tf
        from tensorflow.contrib.slim.python.slim.nets import inception
    except Exception as e:
        GetLog().log().error("code:1003")
    print('Predict data')
    FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
    #print('FLAGS:', FLAGS)
    #print('UNPARSED_ARGV:', UNPARSED_ARGV)

    # Set verbosity
    if FLAGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # using the Winograd non-fused algorithms provides a small performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    #try:
    tf.app.run(main=main, argv=[sys.argv[0:]] + UNPARSED_ARGV)
   # except Exception as e:
   #     GetLog().log().error("code:1002")

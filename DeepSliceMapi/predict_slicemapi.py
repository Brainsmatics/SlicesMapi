from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from PIL import Image
import imageio
import tqdm
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from configparser import ConfigParser
import shutil
import zipfile
from os.path import join, getsize
import logging
import time
from logging.handlers import RotatingFileHandler
import warnings
import math
warnings.filterwarnings("ignore")
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

class GetLog:
    def log(self):
        file_dir = os.getcwd()
        ltime = time.strftime('%Y%m%d', time.localtime(time.time()))
        filename = os.path.join(file_dir, str(ltime) + ".log")
        logger = logging.getLogger("predict_log")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = RotatingFileHandler(filename, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            sh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            sh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger

logger = GetLog().log()


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_enc = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
    return tf.constant(pos_enc, dtype=tf.float32)

def multi_head_attention(q, k, v, num_heads, d_k):
    batch = tf.shape(q)[0]
    seq_len_q = tf.shape(q)[1]
    seq_len_k = tf.shape(k)[1]
    q_proj = slim.fully_connected(q, num_heads * d_k, activation_fn=None)
    k_proj = slim.fully_connected(k, num_heads * d_k, activation_fn=None)
    v_proj = slim.fully_connected(v, num_heads * d_k, activation_fn=None)
    q_split = tf.transpose(tf.reshape(q_proj, [batch, seq_len_q, num_heads, d_k]), [0, 2, 1, 3])
    k_split = tf.transpose(tf.reshape(k_proj, [batch, seq_len_k, num_heads, d_k]), [0, 2, 1, 3])
    v_split = tf.transpose(tf.reshape(v_proj, [batch, seq_len_k, num_heads, d_k]), [0, 2, 1, 3])
    attn_score = tf.matmul(q_split, k_split, transpose_b=True) / math.sqrt(d_k)
    attn_weight = tf.nn.softmax(attn_score)
    attn_out = tf.matmul(attn_weight, v_split)
    attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
    attn_out = tf.reshape(attn_out, [batch, seq_len_q, num_heads * d_k])
    return slim.fully_connected(attn_out, num_heads * d_k, activation_fn=None)

def transformer_encoder_layer(x, d_model, num_heads, d_ff, dropout=0.1, is_training=False):
    d_k = d_model // num_heads
    attn = multi_head_attention(x, x, x, num_heads, d_k)
    attn = slim.dropout(attn, dropout, is_training=is_training)
    x1 = tf.contrib.layers.layer_norm(x + attn)
    ffn = slim.fully_connected(x1, d_ff)
    ffn = slim.fully_connected(ffn, d_model, activation_fn=None)
    ffn = slim.dropout(ffn, dropout, is_training=is_training)
    x2 = tf.contrib.layers.layer_norm(x1 + ffn)
    return x2

# ===================== ResNet+Transformer =====================
def resnet_transformer_regression(image, is_training=False, reuse=tf.AUTO_REUSE,
                                  resnet_depth=50, d_model=256, num_heads=4,
                                  num_layers=2, d_ff=512, dropout_rate=0.1, output_dim=9):
    with tf.variable_scope("resnet_transformer", reuse=reuse):
        img_3ch = tf.tile(image, [1, 1, 1, 3])
        if resnet_depth == 50:
            net, end_points = resnet_v1.resnet_v1_50(img_3ch, is_training=is_training, scope='resnet_v1_50')
        else:
            net, end_points = resnet_v1.resnet_v1_50(img_3ch, is_training=is_training, scope='resnet_v1_50')
        feat_h, feat_w, feat_c = net.shape[1], net.shape[2], net.shape[3]
        seq_len = feat_h * feat_w
        feat_seq = tf.reshape(net, [-1, seq_len, feat_c])
        feat_seq = slim.fully_connected(feat_seq, d_model, activation_fn=None)
        pos_enc = positional_encoding(seq_len, d_model)
        feat_seq = feat_seq + pos_enc[:, :seq_len, :]
        feat_seq = slim.dropout(feat_seq, dropout_rate, is_training=is_training)
        for _ in range(num_layers):
            feat_seq = transformer_encoder_layer(feat_seq, d_model, num_heads, d_ff, dropout_rate, is_training)
        global_feat = tf.reduce_mean(feat_seq, axis=1)
        fc1 = slim.fully_connected(global_feat, 512)
        fc1 = slim.dropout(fc1, dropout_rate, is_training=is_training)
        fc2 = slim.fully_connected(fc1, 256)
        fc2 = slim.dropout(fc2, dropout_rate, is_training=is_training)
        pred = slim.fully_connected(fc2, output_dim, activation_fn=None)
    return pred

def rotation_sitk(fixed_image_sitk, dz, ia, ib, ic):
    ia = np.pi * ia / 180
    ib = np.pi * ib / 180
    ic = np.pi * ic / 180
    [xx, yy, zz] = fixed_image_sitk.GetSize()
    rotation_center = (0, 0, 0)
    rigid_euler = sitk.Euler3DTransform(rotation_center, ia, ib, ic)
    resampled_img = sitk.Resample(fixed_image_sitk, rigid_euler)
    npimage = sitk.GetArrayFromImage(resampled_img)[int(dz), ...]
    return npimage

def resample_sitk_bk(fixed_image_sitk, dz, ia, ib, ic):
    rotation = np.array([np.pi * ia / 180, np.pi * ib / 180, np.pi * ic / 180])
    R = T.euler_matrix(*rotation)[:3, :3]
    [xx, yy, zz] = fixed_image_sitk.GetSize()
    new_origin = (0, 0, 0)
    fixed_image_sitk.SetOrigin(new_origin)
    fixed_image_sitk.SetDirection(np.array(R.flatten()))
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleFilter.SetOutputSpacing([1, 1, 1])
    resampleFilter.SetOutputOrigin(new_origin)
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize((int(xx), int(yy), int(zz)))
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)
    npimage = sitk.GetArrayFromImage(moving_image_sitk)[int(dz), ...]
    return npimage

def matrix_from_anchor_points(AP1, AP2, AP3):
    v1 = AP3 - AP1
    v2 = AP2 - AP1
    n1 = np.cross(v1, v2)
    n2 = np.cross(n1, v1)
    v1_norm = v1 / np.linalg.norm(v1)
    n2_norm = n2 / np.linalg.norm(n2)
    n1_norm = n1 / np.linalg.norm(n1)
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
    zip_name = src_dir + '.zip'
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    z.close()

def resample_sitk(fixed_image_sitk, rx, tx):
    R = np.linalg.inv(rx)
    size = fixed_image_sitk.GetSize()
    spacing = fixed_image_sitk.GetSpacing()
    new_origin = (114, 90, 194) - R.dot(np.array(size) / 2) - R.dot(tx)
    fixed_image_sitk.SetOrigin(new_origin)
    fixed_image_sitk.SetDirection(np.array(R.flatten()))
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin((0, 0, 0))
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize((228, 180, 388))
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)
    npimage = sitk.GetArrayFromImage(moving_image_sitk)[194, ...]
    return npimage

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.float32) / 255.0 * 255.0

def _parse_function_ifind(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'vec': tf.FixedLenFeature([], tf.string),
            'qt': tf.FixedLenFeature([], tf.string),
            'AP1': tf.FixedLenFeature([], tf.string),
            'AP2': tf.FixedLenFeature([], tf.string),
            'AP3': tf.FixedLenFeature([], tf.string)})
    image = tf.reshape(tf.decode_raw(features['image'], tf.float32), [120, 120, 1]) * 255.
    vec = tf.reshape(tf.decode_raw(features['vec'], tf.float32), [3])
    qt = tf.reshape(tf.decode_raw(features['qt'], tf.float32), [4])
    AP1 = tf.reshape(tf.decode_raw(features['AP1'], tf.float32), [3])
    AP2 = tf.reshape(tf.decode_raw(features['AP2'], tf.float32), [3])
    AP3 = tf.reshape(tf.decode_raw(features['AP3'], tf.float32), [3])
    return image, vec, qt, AP1, AP2, AP3

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


ARGPARSER = argparse.ArgumentParser(description='MaskSVRnet Predict ResNet+Transformer')
ARGPARSER.add_argument('--conn_dir', type=str, default='./Config1.ini')
ARGPARSER.add_argument('--model_dir', type=str, default='E:/work/model_train/traindata2/SVRnet/', help='ckpt权重文件夹')
ARGPARSER.add_argument('--data_dir', type=str, default='./test/test1-1.tif', help='测试tif图片路径')
ARGPARSER.add_argument('--save_dir', type=str, default='./predict_result/1_', help='预测输出保存路径')
ARGPARSER.add_argument('--log_dir', type=str, default='./predict_result/1_')
ARGPARSER.add_argument('--subject_id', type=str, default='fixed_mask2_25')
ARGPARSER.add_argument('--n_iter', type=int, default=100)
ARGPARSER.add_argument('--gpu', type=str, default='0', help='指定GPU')
ARGPARSER.add_argument('--debug', default=False, action='store_true')

def main(argv):
    # GPU指定
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    conn_dir = FLAGS.conn_dir
    conn = ConfigParser()
    try:
        conn.read(conn_dir, encoding="utf-8")
    except Exception as e:
        logger.error(f"读取配置文件失败: {str(e)} code:404")
        return

    data_path = FLAGS.data_dir
    savedir = FLAGS.save_dir
    model_dir = FLAGS.model_dir
    os.makedirs(savedir, exist_ok=True)

    # 读取固定Nifti模板
    try:
        fixed_path = conn.get('DIR', 'fixed_dir')
        fixed_image_sitk_tmp = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        fixed_image_sitk = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
        ann_path = conn.get('DIR', 'ann_dir')
        ann_image_sitk_tmp = sitk.ReadImage(ann_path, sitk.sitkFloat32)
        ann_image_sitk = sitk.GetImageFromArray(sitk.GetArrayFromImage(ann_image_sitk_tmp))
    except Exception as e:
        logger.error(f"读取固定模板Nifti失败: {str(e)} code:404")
        return

    # 1. 输入占位
    input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 1], name="input_image")
    # 2. ResNet+Transformer推理网络
    y_pred = resnet_transformer_regression(input_ph, is_training=False)
    AP1_pred, AP2_pred, AP3_pred = tf.split(y_pred, 3, axis=1)

    # 3. 读取本地tif测试图
    try:
        image1 = Image.open(data_path).convert("L")
        image1 = image1.resize([224, 224])
        image_np = load_image_into_numpy_array(image1)
        image_np_expanded = np.expand_dims(image_np, axis=0)
    except Exception as e:
        logger.error(f"读取测试图片失败: {str(e)}")
        return

    # 4. Session加载权重
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    ckpt_file = tf.train.latest_checkpoint(model_dir)
    if ckpt_file is None:
        logger.error("未找到任何checkpoint文件！")
        return
    saver.restore(sess, ckpt_file)
    logger.info(f'成功加载权重: {ckpt_file}')

    # 5. 前向推理预测AP点
    pred_ap1, pred_ap2, pred_ap3 = sess.run([AP1_pred, AP2_pred, AP3_pred], feed_dict={input_ph: image_np_expanded})
    pred_ap1 = pred_ap1[0]
    pred_ap2 = pred_ap2[0]
    pred_ap3 = pred_ap3[0]

    # 6. 构造旋转矩阵+重采样图像
    try:
        rx = matrix_from_anchor_points(pred_ap1, pred_ap2, pred_ap3)
        tx = pred_ap2 * 194
        fixed_pred = resample_sitk(fixed_image_sitk, rx, tx)
    except Exception as e:
        logger.error(f"生成旋转矩阵/重采样失败: {str(e)} code:1001")
        return

    # 7. 保存结果、角度参数、打包zip
    try:
        imageio.imsave(os.path.join(savedir, 'fixed.tif'), np.uint16(fixed_pred))
        imageio.imsave(os.path.join(savedir, 'show.jpg'), np.uint8(fixed_pred))
        Angles = rotationMatrixToAngles(rx)
        with open(os.path.join(savedir, 'parameter.txt'), 'w', encoding="utf-8") as f:
            f.write(f"旋转欧拉角(弧度): {str(Angles)}\n")
            f.write(f"平移tx: {str(tx)}\n")
        # 打包
        zip_path = os.path.join(savedir, 'result.zip')
        zfile = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
        zfile.write(os.path.join(savedir, 'fixed.tif'), "fixed.tif")
        zfile.write(os.path.join(savedir, 'show.jpg'), "show.jpg")
        zfile.write(os.path.join(savedir, 'parameter.txt'), "parameter.txt")
        zfile.close()
        logger.info("预测完成，结果保存成功 code:200")
    except Exception as e:
        logger.error(f"保存输出文件失败: {str(e)} code:1002")
        return

if __name__ == '__main__':

    if FLAGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    logger.info("=====  ResNet+Transformer  =====")
    try:
        tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED_ARGV)
    except Exception as e:
        logger.error(f"程序全局异常: {str(e)} code:1003")
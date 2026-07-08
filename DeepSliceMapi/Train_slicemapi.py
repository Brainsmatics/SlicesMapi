from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import math

import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train ResNet+Transformer Regression for Anchor Points')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='E:/work/model_train/traindata2/',
    help='The path to the dataset directory.')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='E:/work/model_train/traindata2/SVRnet/',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--params', type=str, default='E:/work/model_train/traindata2/SVRnet/param.json',
    help='Auxiliary parameters file.')
# training parameters
ARGPARSER.add_argument(
    '--train_iter', type=int, default=10000,
    help='The number of training steps.')
ARGPARSER.add_argument(
    '--init_lr', type=float, default=1e-4,
    help='Initial Learning rate.')
# snapshot parameters
ARGPARSER.add_argument(
    '--ckpt_steps', type=int, default=100,
    help='Number of steps between checkpoint saves.')
# memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=64,
    help='The number of data points per batch.')
ARGPARSER.add_argument(
    '--memcap', type=float, default=1.0,
    help='Maximum fraction of memory to allocate per GPU.')
# data loading
ARGPARSER.add_argument(
    '--queue_threads', type=int, default=8,
    help='How many parallel threads to run for data queuing.')
ARGPARSER.add_argument(
    '--queue_buffer', type=int, default=1000,
    help='How many samples to queue up.')
# logging
ARGPARSER.add_argument(
    '--log_steps', type=int, default=10,
    help='Global steps between log output.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard output.")
ARGPARSER.add_argument(
    '--initial_eval', default=False, action='store_true',
    help="Runs an evaluation before the first training iteration.")
# multi-gpu systems
ARGPARSER.add_argument(
    '--gpu', type=str, default='0',
    help='Specify default GPU to use.')

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

    image   = tf.reshape(tf.decode_raw(features['image'], tf.float32),[180,228,1]) * 255.
    vec     = tf.reshape(tf.decode_raw(features['vec'], tf.float32),[3])
    qt      = tf.reshape(tf.decode_raw(features['qt'], tf.float32),[4])
    AP1     = tf.reshape(tf.decode_raw(features['AP1'], tf.float32),[3])
    AP2     = tf.reshape(tf.decode_raw(features['AP2'], tf.float32),[3])
    AP3     = tf.reshape(tf.decode_raw(features['AP3'], tf.float32),[3])

    return image, vec, qt, AP1, AP2, AP3

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_enc = np.concatenate([sines, cosines], axis=-1)
    pos_enc = pos_enc[np.newaxis, ...]
    return tf.constant(pos_enc, dtype=tf.float32)

def multi_head_attention(q, k, v, num_heads, d_k):
    batch = tf.shape(q)[0]
    seq_len_q = tf.shape(q)[1]
    seq_len_k = tf.shape(k)[1]

    q_proj = slim.fully_connected(q, num_heads * d_k, activation_fn=None)
    k_proj = slim.fully_connected(k, num_heads * d_k, activation_fn=None)
    v_proj = slim.fully_connected(v, num_heads * d_k, activation_fn=None)

    # [B, H, L, dk]
    q_split = tf.transpose(tf.reshape(q_proj, [batch, seq_len_q, num_heads, d_k]), [0,2,1,3])
    k_split = tf.transpose(tf.reshape(k_proj, [batch, seq_len_k, num_heads, d_k]), [0,2,1,3])
    v_split = tf.transpose(tf.reshape(v_proj, [batch, seq_len_k, num_heads, d_k]), [0,2,1,3])

    # Scaled dot product attention
    attn_score = tf.matmul(q_split, k_split, transpose_b=True) / math.sqrt(d_k)
    attn_weight = tf.nn.softmax(attn_score)
    attn_out = tf.matmul(attn_weight, v_split)

    # concat heads
    attn_out = tf.transpose(attn_out, [0,2,1,3])
    attn_out = tf.reshape(attn_out, [batch, seq_len_q, num_heads * d_k])
    out = slim.fully_connected(attn_out, num_heads * d_k, activation_fn=None)
    return out

def transformer_encoder_layer(x, d_model, num_heads, d_ff, dropout=0.1):
    d_k = d_model // num_heads
    # Multi-head self attention
    attn = multi_head_attention(x, x, x, num_heads, d_k)
    attn = slim.dropout(attn, dropout)
    x1 = tf.contrib.layers.layer_norm(x + attn)

    # FFN
    ffn = slim.fully_connected(x1, d_ff)
    ffn = slim.fully_connected(ffn, d_model, activation_fn=None)
    ffn = slim.dropout(ffn, dropout)
    x2 = tf.contrib.layers.layer_norm(x1 + ffn)
    return x2

def resnet_transformer_regression(image,
                                  resnet_depth=50,
                                  d_model=256,
                                  num_heads=4,
                                  num_layers=2,
                                  d_ff=512,
                                  dropout_rate=0.1,
                                  output_dim=9):
    """
    image: [B,224,224,1] 
    return: pred [B,9]
    """
    img_3ch = tf.tile(image, [1,1,1,3])

    # ResNet V1 
    if resnet_depth == 50:
        net, end_points = resnet_v1.resnet_v1_50(img_3ch, is_training=True, scope='resnet_v1_50')
    elif resnet_depth == 101:
        net, end_points = resnet_v1.resnet_v1_101(img_3ch, is_training=True, scope='resnet_v1_101')
    else:
        net, end_points = resnet_v1.resnet_v1_50(img_3ch, is_training=True, scope='resnet_v1_50')

    feat_h, feat_w, feat_c = net.shape[1], net.shape[2], net.shape[3]
    seq_len = feat_h * feat_w
    feat_seq = tf.reshape(net, [-1, seq_len, feat_c])

    feat_seq = slim.fully_connected(feat_seq, d_model, activation_fn=None)

    pos_enc = positional_encoding(seq_len, d_model)
    feat_seq = feat_seq + pos_enc[:, :seq_len, :]
    feat_seq = slim.dropout(feat_seq, dropout_rate)

    for _ in range(num_layers):
        feat_seq = transformer_encoder_layer(feat_seq, d_model, num_heads, d_ff, dropout_rate)


    global_feat = tf.reduce_mean(feat_seq, axis=1)

    fc1 = slim.fully_connected(global_feat, 512)
    fc1 = slim.dropout(fc1, dropout_rate)
    fc2 = slim.fully_connected(fc1, 256)
    fc2 = slim.dropout(fc2, dropout_rate)
    pred = slim.fully_connected(fc2, output_dim, activation_fn=None)
    return pred

def main(argv):
    tf.reset_default_graph()
    # 1. TFRecord Dataset 
    files = tf.data.Dataset.list_files(tf.gfile.Glob(FLAGS.data_dir + '*.tfrecord'))
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
    dataset = dataset.map(_parse_function_ifind, num_parallel_calls=FLAGS.queue_threads)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(FLAGS.queue_buffer)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    image, vec, qt, AP1, AP2, AP3 = iterator.get_next()


    image = tf.image.resize_images(image, size=[224, 224])

    # 2. ResNet + Transformer 
    y_pred = resnet_transformer_regression(image, output_dim=9)
    AP1_pred, AP2_pred, AP3_pred = tf.split(y_pred, 3, axis=1)

    # 3. L2 loss
    l1 = tf.nn.l2_loss(AP1_pred - AP1, name='loss/l1')
    l2 = tf.nn.l2_loss(AP2_pred - AP2, name='loss/l2')
    l3 = tf.nn.l2_loss(AP3_pred - AP3, name='loss/l3')

    tf.summary.scalar('AnchorPoints_loss/AP1', l1)
    tf.summary.scalar('AnchorPoints_loss/AP2', l2)
    tf.summary.scalar('AnchorPoints_loss/AP3', l3)
    total_loss = l1 + l2 + l3
    tf.summary.scalar('loss', total_loss)

    # 4. Optimizer
    train_op = tf.train.AdamOptimizer(FLAGS.init_lr).minimize(total_loss)

    # 5. Session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memcap

    saver = tf.train.Saver(max_to_keep=10)
    os.makedirs(FLAGS.model_dir, exist_ok=True)

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, 'train'), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(FLAGS.train_iter):
                start_time = time.time()
                _, summary, loss_val = sess.run([train_op, merged_summary, total_loss])

                train_writer.add_summary(summary, step)
                duration = time.time() - start_time

                if step % FLAGS.log_steps == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    log_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f sec/batch)')
                    print(log_str % (datetime.now(), step, loss_val, examples_per_sec, duration))

                if step % FLAGS.ckpt_steps == 0:
                    tf.logging.info('Saving checkpoint step: {}'.format(step))
                    save_path = os.path.join(FLAGS.model_dir, 'iter_{}.ckpt'.format(step))
                    saver.save(sess, save_path)

        except tf.errors.OutOfRangeError:
            tf.logging.info('Training dataset exhausted')
        except KeyboardInterrupt:
            tf.logging.info('Keyboard interrupt received')
        finally:
            tf.logging.info('Shutting down queue threads')
            coord.request_stop()
            coord.join(threads)
            final_save = os.path.join(FLAGS.model_dir, 'iter_{}.ckpt'.format(step))
            saver.save(sess, final_save)
            tf.logging.info('Final model saved at {}'.format(final_save))

if __name__ == '__main__':
    print('Training ResNet+Transformer Anchor Point Regression Model (TF1 slim)')
    FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
    print('FLAGS:', FLAGS)
    print('UNPARSED_ARGV:', UNPARSED_ARGV)

    # Log level
    if FLAGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU control
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED_ARGV)
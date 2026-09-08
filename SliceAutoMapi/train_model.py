from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim import arg_scope

# parameters
RAW_SHAPE = (299, 299, 1)  # train slices
TARGET_SIZE = (224, 224)
PATCH_NUM = 4
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
DROPOUT_RATE = 0.1
OUTPUT_DIM = 9

BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 100
SHUFFLE_BUFFER = 10000
PREFETCH_BUFFER = 4

DATA_DIR = r""  # trainsets 
CKPT_SAVE_DIR = "./checkpoint" #model save dir
PRETRAIN_RESNET = None

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


def main():
    dataset = build_dataset()
    iterator = dataset.make_one_shot_iterator()
    img_batch, label_batch = iterator.get_next()

    coarse_pred, fine_pred = build_model(img_batch, is_training=True)

    loss_coarse = tf.losses.mean_squared_error(labels=label_batch, predictions=coarse_pred)
    loss_fine = tf.losses.mean_squared_error(labels=label_batch, predictions=fine_pred)
    total_loss = loss_coarse + loss_fine

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    clipped_grads = [(tf.clip_by_norm(g, 1.0), v) for g, v in grads_and_vars if g is not None]
    train_op = optimizer.apply_gradients(clipped_grads)

    print_op = tf.print(
        "step loss_coarse:", loss_coarse,
        " loss_fine:", loss_fine,
        " total_loss:", total_loss,
        " lr:", LEARNING_RATE
    )

    saver = tf.train.Saver(max_to_keep=5)
    os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if PRETRAIN_RESNET is not None and os.path.exists(PRETRAIN_RESNET):
            saver_resnet = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1_50"))
            saver_resnet.restore(sess, PRETRAIN_RESNET)
            print("load resnet pretrain done")

        step = 0
        while True:
            try:
                _, loss_val, _ = sess.run([train_op, total_loss, print_op])
                step += 1
                if step % 100 == 0:
                    saver.save(sess, os.path.join(CKPT_SAVE_DIR, "model.ckpt"), global_step=step)
                    print(f"save ckpt step={step}")
            except tf.errors.OutOfRangeError:
                print("training finished!")
                break


if __name__ == "__main__":
    main()
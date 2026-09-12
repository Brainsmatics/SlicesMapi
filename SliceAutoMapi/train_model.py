from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim import arg_scope

# ====================== 超参数 ======================
RAW_SHAPE = (299, 299, 1)
TARGET_SIZE = (224, 224)
PATCH_NUM = 4
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
DROPOUT_RATE = 0.1
OUTPUT_DIM = 9

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 100
SHUFFLE_BUFFER = 10000
PREFETCH_BUFFER = 4

DATA_DIR = r"H:/SVR/"
CKPT_SAVE_DIR = "./checkpoint5"
SAVE_DIR = "./checkpoint5/"
PRETRAIN_RESNET = None

# ====================== 工具函数 ======================
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

def split_points(v):
    return v[:,0:3],v[:,3:6],v[:,6:9]

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


# ===================== 【关键】封装resnet特征提取，使用AUTO_REUSE共享权重 =====================
def resnet_extract_feat(input_img, is_training):
    with tf.variable_scope("resnet_v1_50", reuse=tf.AUTO_REUSE):
        with arg_scope(resnet_v1.resnet_arg_scope()):
            net, _ = resnet_v1.resnet_v1_50(input_img, num_classes=None, is_training=is_training)
    feat = tf.squeeze(net, axis=[1, 2])
    return feat




def build_model(image_batch, is_training=True):
    image_resized = tf.image.resize_images(image_batch, TARGET_SIZE)

    with tf.variable_scope("coarse_branch"):
        feat_global = resnet_extract_feat(image_resized, is_training)
        coarse_densel = tf.layers.dense(feat_global, 256, activation=tf.nn.relu)
        coarse_anchor = tf.layers.dense(coarse_densel, OUTPUT_DIM, kernel_initializer=tf.truncated_normal_initializer())

    with tf.variable_scope("transformer_fine_branch"):
        # global token
        global_token = tf.layers.dense(feat_global, EMBED_DIM)
        global_token = tf.expand_dims(global_token, axis=1)

        # coarse坐标编码成token送入Transformer（推荐保留）
        coarse_emb = tf.layers.dense(coarse_anchor, EMBED_DIM)
        coarse_token = tf.expand_dims(coarse_emb, axis=1)

        # patch提取
        patch_stack = tf.map_fn(split_image_to_patches, image_batch, dtype=tf.float32)
        patch_feats = []
        for i in range(PATCH_NUM):
            patch_img = patch_stack[:, i, ...]
            patch_feat = resnet_extract_feat(patch_img, is_training)
            patch_feat = tf.layers.dense(patch_feat, EMBED_DIM)
            patch_feats.append(patch_feat)
        patch_tokens = tf.stack(patch_feats, axis=1)

        # token拼接：global + coarse + patches
        seq_tokens = tf.concat([global_token, coarse_token, patch_tokens], axis=1)
        pos_emb = tf.get_variable("pos_emb", shape=[1, PATCH_NUM+2, EMBED_DIM], dtype=tf.float32)
        seq_tokens = seq_tokens + pos_emb

        # transformer block
        for _ in range(NUM_LAYERS):
            seq_tokens = transformer_block(seq_tokens, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)

        fused_feat = seq_tokens[:, 0, :]
        fine_dense = tf.layers.dense(fused_feat, 256, activation=tf.nn.relu)
        fine_anchor = tf.layers.dense(fine_dense, OUTPUT_DIM)
    return coarse_anchor, fine_anchor


# ====================== 构建数据集 pipeline（shuffle/repeat/prefetch） ======================
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

    #coarse loss
    c1,c2,c3 = split_points(coarse_pred)
    gt1,gt2,gt3 =split_points(label_batch)
    loss_c = tf.reduce_mean(tf.norm(c1-gt1,axis=1)+tf.norm(c2-gt2,axis=1)+tf.norm(c3-gt3,axis=1))

    #fine loss
    f1,f2,f3 = split_points(fine_pred)
    loss_f = tf.reduce_mean(tf.norm(f1-gt1,axis=1)+tf.norm(f2-gt2,axis=1)+tf.norm(f3-gt3,axis=1))

    # margin损失：fine误差大于coarse时才惩罚
    margin = tf.maximum(0.0, loss_f - loss_c)
    margin_loss = tf.reduce_mean(margin)

    # 修正成功样本占比（监控指标）
    success_mask = tf.cast(loss_f < loss_c, tf.float32)
    success_ratio = tf.reduce_mean(success_mask)
    


    
    #merge loss
    total_loss = 0.4 * loss_c + 1.0 * loss_f + 0.2 * margin_loss

    all_vars = tf.trainable_variables()
    coarse_vars = [v for v in all_vars if v.name.startswith("coarse_branch/")]
    trans_fine_vars = [v for v in all_vars if v.name.startswith("transformer_fine_branch/")]

    # ========= Stage1：只更新 coarse_vars，lr=1e-4 =========
    opt_stage1 = tf.train.AdamOptimizer(learning_rate=1e-4)
    # compute_gradients 指定var_list，只对coarse变量求导
    grads_vars_stage1 = opt_stage1.compute_gradients(total_loss, var_list=coarse_vars)
    # 梯度裁剪，过滤None梯度
    clipped_grads_stage1 = [(tf.clip_by_norm(g, 1.0), v) for g, v in grads_vars_stage1 if g is not None]
    apply_grad_stage1 = opt_stage1.apply_gradients(clipped_grads_stage1)
    # BN更新ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op_stage1 = tf.group([apply_grad_stage1] + update_ops)


    # ========= Stage2：更新 coarse + transformer_fine，lr=3e-5 =========
    opt_stage2 = tf.train.AdamOptimizer(learning_rate=3e-5)
    trainable_stage2_vars = coarse_vars + trans_fine_vars
    grads_vars_stage2 = opt_stage2.compute_gradients(total_loss, var_list=trainable_stage2_vars)
    clipped_grads_stage2 = [(tf.clip_by_norm(g, 1.0), v) for g, v in grads_vars_stage2 if g is not None]
    apply_grad_stage2 = opt_stage2.apply_gradients(clipped_grads_stage2)
    train_op_stage2 = tf.group([apply_grad_stage2] + update_ops)
    
    print_op = tf.print(
        "step loss_coarse:", loss_c,
        " loss_fine:", loss_f,
        " total_loss:", total_loss,
        " lr:", LEARNING_RATE
    )

    saver = tf.train.Saver(max_to_keep=10)
    os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

    # ========= plateau检测变量（Python侧，放这里） =========
    loss_history = []
    window_size = 50
    min_delta = 0.001
    plateau_count = 0
    plateau_trigger = 3

    def check_loss_plateau(current_loss):
        global plateau_count
        loss_history.append(current_loss)
        if len(loss_history) > window_size:
            loss_history.pop(0)
        if len(loss_history) < window_size:
            plateau_count = 0
            return False
        loss_start = loss_history[0]
        loss_end = loss_history[-1]
        delta = loss_start - loss_end
        if delta < min_delta:
            plateau_count += 1
            print(f"plateau count: {plateau_count}, delta:{delta:.4f}")
            if plateau_count >= plateau_trigger:
                return True
        else:
            plateau_count = 0
        return False
    # ==============================================

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if PRETRAIN_RESNET is not None and os.path.exists(PRETRAIN_RESNET):
            saver_resnet = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="coarse_branch"))
            saver_resnet.restore(sess, PRETRAIN_RESNET)
            print("load resnet pretrain done")

        step = 0
        stage = 1  # 新增阶段标记
        max_stage1_step = 10000 # stage1最大步数兜底

        while True:
            try:
                if stage == 1:
                    # ========= Stage1 只训练coarse =========
                    _, lc_val = sess.run([train_op_stage1, loss_c])
                    step += 1

                    # 判断是否到达平台，切换stage2
                    if check_loss_plateau(lc_val):
                        saver.save(sess, os.path.join(CKPT_SAVE_DIR, "stage1_coarse_only.ckpt"), global_step=step)
                        stage = 2
                        print("==== Switch to stage2, open transformer & fine head ====")
                    # 兜底强制切换
                    if step > max_stage1_step:
                        saver.save(sess, os.path.join(CKPT_SAVE_DIR, "stage1_force.ckpt"), global_step=step)
                        stage = 2
                        print("Force switch stage2, reach max stage1 step")

                    # 每10步打印（stage1只打印coarse loss）
                    if step % 10 == 0:
                        lc_val = sess.run(loss_c)
                        print(f"step:{step} | stage1 loss_c:{lc_val:.4f}")

                else:
                    # ========= Stage2 全部参数训练，包含margin loss =========
                    _, lc_val, lf_val, lm_val, succ_ratio_val = sess.run([
                        train_op_stage2, loss_c, loss_f, margin_loss, success_ratio
                    ])
                    step +=1

                    if step % 10 == 0:
                        print(f"step:{step} | stage2 loss_c:{lc_val:.4f}, loss_f:{lf_val:.4f}, margin:{lm_val:.4f}, success_rate:{succ_ratio_val:.3f}")

                # 每100步保存模型
                if step % 100 == 0:
                    saver.save(sess, os.path.join(SAVE_DIR, "model.ckpt"), global_step=step)

                # 【你原来500步验证代码，保持原样放在这里】
                # if step %500 ==0:
                #    run validation ...

            except tf.errors.OutOfRangeError:
                print("epoch end")
                break



if __name__ == "__main__":
    main()
import numpy as np
from VaeCelebA.model_vae import encoder, generator
from VaeCelebA.train_vae import FLAGS
import tensorflow as tf
import tensorlayer as tl
from VaeCelebA.utils import *
from glob import glob
import os
import cv2
import scipy.misc


input_imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size,
                                         FLAGS.output_size, FLAGS.c_dim], name='real_images')

# 标准正太分布z_p
z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
# 用于重参数技巧的eps
eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
# vae学习率
lr_vae = tf.placeholder(tf.float32, shape=[])

# ----------------------encoder----------------------
net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=False, reuse=False)

# ----------------------decoder----------------------
# decode z
# z = z_mean + z_sigma * eps
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))  # 重参数技巧
gen0, _ = generator(z, is_train=False, reuse=False)

# 加载预训练模型
sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)
load_e_params = tl.files.load_npz(path=r"/VaeCelebA\checkpoint\vae_0808",
                                  name='/net_e.npz')
tl.files.assign_params(sess, load_e_params[:24], net_out1)
net_out1.print_params(False)
tl.files.assign_params(sess, np.concatenate((load_e_params[:24], load_e_params[30:]), axis=0), net_out2)
net_out2.print_params(False)

load_g_params = tl.files.load_npz(path=r"/VaeCelebA\checkpoint\vae_0808", name='/net_g.npz')
tl.files.assign_params(sess, load_g_params, gen0)
gen0.print_params(False)

# 读取图片
data_files = glob(os.path.join(r"/VaeCelebA\data", FLAGS.dataset, "*.jpg"))
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches

# 生成mini_batch迭代器, 一个batch有64张图片
minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=FLAGS.batch_size,
                                   shuffle=True)

cnt = 0      # 计数器

while cnt < 50:          # 迭代50次
    print("iter"+str(cnt+1)+"...")
    batch_files, _ = minibatch.__next__()

    input_image = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                             is_grayscale=0) for batch_file in batch_files]
    for id, batch_file in enumerate(batch_files):
        img = cv2.imread(batch_file)
        cv2.imwrite("D:/vaeInterpretable/VaeCelebA/VaeCelebA/data/test_img/"+str(64*cnt+id+1)+".jpg", img)

    input_image = np.array(input_image).astype(np.float32)
    intent_variable = sess.run(z, feed_dict={input_imgs: input_image})     # intent_variable.shape=(64, 40)

    for i in range(64):                                # 每一个隐变量扩展生成64个隐变量送入ecoder生成64张图片
        tmp = intent_variable[i]
        decoder_input = np.zeros((64, 40))
        val = -abs(3 * np.random.randn(40))
        for m in range(40):                           # 起始值下限是-3
            if val[m] < -3:
                val[m] = -3
        val[12] = tmp[12]                             # 固定12号特征
        for k in range(64):
            for kk in range(40):                       # 控制其他所有特征按照特定的规律进行变化
                decoder_input[k][kk] = val[kk]
            delta = 0.1*abs(np.random.randn(40))
            for ii in range(40):                       # 增量上限为0.1
                if delta[ii] > 0.1:
                    delta[ii] = 0.1
            delta[12] = 0
            val = val + delta
        img_result = sess.run(gen0.outputs, feed_dict={z: decoder_input})
        for j in range(64):
            scipy.misc.imsave("D:/vaeInterpretable/VaeCelebA/VaeCelebA/data/img_save_1/"+str(cnt*4096+i*64+j+1)+".png", img_result[j])
    cnt = cnt + 1
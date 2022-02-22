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

load_e_params = tl.files.load_npz(path=r"D:\vaeInterpretable\vae-celebA-master\VaeCelebA\checkpoint",
                                  name=r'\net_e.npz')
tl.files.assign_params(sess, load_e_params[:24], net_out1)
net_out1.print_params(False)
tl.files.assign_params(sess, np.concatenate((load_e_params[:24], load_e_params[30:]), axis=0), net_out2)
net_out2.print_params(False)

load_g_params = tl.files.load_npz(path=r"D:\vaeInterpretable\vae-celebA-master\VaeCelebA\checkpoint", name=r'\net_g.npz')
tl.files.assign_params(sess, load_g_params, gen0)
gen0.print_params(False)


data_files = glob(os.path.join(r"D:\vaeInterpretable\vae-celebA-master\VaeCelebA\data", FLAGS.dataset, "*.jpg"))
data_files = sorted(data_files)
data_files = np.array(data_files)  # for tl.iterate.minibatches
print(data_files.shape)
minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=FLAGS.batch_size,
                                   shuffle=True)

batch_files, _ = minibatch.__next__()
img = cv2.imread(batch_files[0])
cv2.imwrite(r'D:\vaeInterpretable\vae-celebA-master\VaeCelebA\sample.png', img)

input_image = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                         is_grayscale=0) for batch_file in batch_files]

input_image = np.array(input_image).astype(np.float32)

intent_variable = sess.run(z, feed_dict={input_imgs: input_image})

print(intent_variable.shape)

# with open(r"D:\vaeInterpretable\VaeCelebA\VaeCelebA\kl_dim.txt", 'w') as f:
#     for i in range(40):
#         f.write("dim"+str(i)+'\t')
#     f.write('\n')
#     for i in range(64):
#         for j in range(40):
#             f.write(str(intent_variable[i][j])+'\t')
#         f.write('\n')

tmp = intent_variable[0]
for i in range(64):
    intent_variable[i] = tmp

val = -2
for i in range(64):
    intent_variable[i][16] = val
    val = val + 6/64

with open('./test', 'w') as f:
    for item in intent_variable:
        for num in item:
            f.write(str(num)+'\t')
        f.write('\n')

img_result = sess.run(gen0.outputs, feed_dict={z: intent_variable})

save_images(img_result, [8,8], r"D:\vaeInterpretable\vae-celebA-master\VaeCelebA\sample_result.png")

# for i in range(64):
#     scipy.misc.imsave("D:/vaeInterpretable/VaeCelebA/VaeCelebA/img_save/"+str(i)+".png", img_result[i])


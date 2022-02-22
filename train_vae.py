import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import math
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from VaeCelebA.model_vae import  *
from VaeCelebA.utils import *
import matplotlib.pyplot as plt
import heapq
import matplotlib

pp = pprint.PrettyPrinter()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
Tensorlayer implementation of VAE
'''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 3, "Epoch to train [5]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [108]")
# flags.DEFINE_integer("decoder_output_size", 64, "The size of the output images to produce from decoder[64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 40, "Dimension of latent representation vector from. [2048]")
flags.DEFINE_integer("sample_step", 300, "The interval of generating sample. [300]")
flags.DEFINE_integer("save_step", 800, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "img_celeba", "The name of dataset [celebA]")
flags.DEFINE_string("test_number", "vae_0307", "The number of experiment [test2]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
# flags.DEFINE_integer("class_dim", 4, "class number for auxiliary classifier [5]") 
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("load_pretrain", False, "Default to False;If start training on a pretrained net, choose True")
FLAGS = flags.FLAGS


def main(_):
    global kl_dim
    pp.pprint(FLAGS.__flags)

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    ##========================= DEFINE MODEL ===========================##
    # the input_imgs are input for both encoder and discriminator
    input_imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size,
                                             FLAGS.output_size, FLAGS.c_dim], name='real_images')

    # 标准正太分布z_p
    z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
    # 用于重参数技巧的eps
    eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
    # vae学习率
    lr_vae = tf.placeholder(tf.float32, shape=[])

    # ----------------------encoder----------------------
    # z_mean (64, 40), z_log_sigma_sq (64, 40)
    net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False)

    # ----------------------decoder----------------------
    # decode z
    # z = z_mean + z_sigma * eps
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))  # 重参数技巧
    gen0, _ = generator(z, is_train=True, reuse=False)

    # ----------------------for samples----------------------
    gen2, gen2_logits = generator(z, is_train=False, reuse=True)
    gen3, gen3_logits = generator(z_p, is_train=False, reuse=True)

    ##========================= DEFINE TRAIN OPS =======================##
    ''''
    reconstruction loss:
    use the pixel-wise mean square error in image space
    '''
    # 重构损失：均方误差
    SSE_loss = tf.reduce_mean(tf.square(gen0.outputs - input_imgs))  # /FLAGS.output_size/FLAGS.output_size/3
    '''
    KL divergence:
    we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
    then compute KL divergence between z and standard normal gaussian N(0,I) 
    '''
    # KL散度
    KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1))
    KL_item = - 0.5 * (1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq))  # (64, 40)
    KL_dim = tf.reduce_mean(KL_item, axis=0)                              # (1, 40)
    ### important points! ###
    # the weight between style loss(KLD) and contend loss(pixel-wise mean square error)
    VAE_loss = 0.005 * KL_loss + SSE_loss  # KL_loss isn't working well if the weight of SSE is too big

    e_vars = tl.layers.get_variables_with_name('encoder', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    # d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    vae_vars = e_vars + g_vars

    print("-------encoder-------")
    net_out1.print_params(False)
    print("-------generator-------")
    gen0.print_params(False)

    # optimizers for updating encoder, discriminator and generator
    vae_optim = tf.train.AdamOptimizer(lr_vae, beta1=FLAGS.beta1) \
        .minimize(VAE_loss, var_list=vae_vars)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    # prepare file under checkpoint_dir
    model_dir = "vae_0307"
    #  there can be many models under one checkpoine file
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)  # './checkpoint/vae_0808'
    tl.files.exists_or_mkdir(save_dir)
    # under current directory
    samples_1 = FLAGS.sample_dir + "/" + FLAGS.test_number
    # samples_1 = FLAGS.sample_dir + "/test2"
    tl.files.exists_or_mkdir(samples_1)

    # 加载模型继续训练
    if FLAGS.load_pretrain == True:
        load_e_params = tl.files.load_npz(path=save_dir, name='/net_e.npz')
        tl.files.assign_params(sess, load_e_params[:24], net_out1)
        net_out1.print_params(True)
        tl.files.assign_params(sess, np.concatenate((load_e_params[:24], load_e_params[30:]), axis=0), net_out2)
        net_out2.print_params(True)

        load_g_params = tl.files.load_npz(path=save_dir, name='/net_g.npz')
        tl.files.assign_params(sess, load_g_params, gen0)
        gen0.print_params(True)

    # get the list of absolute paths of all images in dataset
    data_files = glob(os.path.join("D:/vaeInterpretable/vae-celebA-master/VaeCelebA/data", FLAGS.dataset, "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches


    ##========================= TRAIN MODELS ================================##
    iter_counter = 0

    training_start_time = time.time()
    # use all images in dataset in every epoch
    kl_each_dim = [[] for i in range(FLAGS.z_dim)]
    for epoch in range(FLAGS.epoch):
        ## shuffle data
        print("[*] Dataset shuffled!")
        minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=FLAGS.batch_size,
                                           shuffle=True)
        idx = 0
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
        kl_dim_batch = []
        while True:
            try:
                batch_files, _ = minibatch.__next__()
                batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size,
                                   is_grayscale=0) \
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                start_time = time.time()
                vae_current_lr = FLAGS.learning_rate
                # update
                kl, kl_dim, sse, errE, _ = sess.run([KL_loss, KL_dim, SSE_loss, VAE_loss, vae_optim],
                                                    feed_dict={input_imgs: batch_images, lr_vae: vae_current_lr})
                kl_dim_batch.append(kl_dim)
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, vae_loss:%.8f, kl_loss:%.8f, sse_loss:%.8f" \
                      % (epoch, FLAGS.epoch, idx, batch_idxs,
                         time.time() - start_time, errE, kl, sse))
                sys.stdout.flush()

                iter_counter += 1
                # save samples
                if np.mod(iter_counter, FLAGS.sample_step) == 0:
                    # generate and visualize generated images
                    img1, img2 = sess.run([gen2.outputs, gen3.outputs], feed_dict={input_imgs: batch_images})
                    save_images(img1, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(samples_1, epoch, idx))

                    # img2 = sess.run(gen3.outputs, feed_dict={input_imgs: batch_images})
                    save_images(img2, [8, 8],
                                './{}/train_{:02d}_{:04d}_random.png'.format(samples_1, epoch, idx))

                    # save input image for comparison
                    save_images(batch_images, [8, 8], './{}/input.png'.format(samples_1))
                    print("[Sample] sample generated!!!")
                    sys.stdout.flush()

                # save checkpoint
                if np.mod(iter_counter, FLAGS.save_step) == 0:
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    net_e_name = os.path.join(save_dir, 'net_e.npz')
                    net_g_name = os.path.join(save_dir, 'net_g.npz')
                    # this version is for future re-check and visualization analysis
                    net_e_iter_name = os.path.join(save_dir, 'net_e_%d.npz' % iter_counter)
                    net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)

                    # params of two branches
                    net_out_params = net_out1.all_params + net_out2.all_params
                    # remove repeat params
                    net_out_params = tl.layers.list_remove_repeat(net_out_params)
                    tl.files.save_npz(net_out_params, name=net_e_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_name, sess=sess)

                    tl.files.save_npz(net_out_params, name=net_e_iter_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_iter_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")

                idx += 1
                # print idx
            except StopIteration:
                print('>>>>>>>>one epoch finished...')
                kl_dim = np.mean(kl_dim_batch, axis=0)              # 在一个epoch结束时计算每一个batch各个维度kl散度的平均值
                for i in range(FLAGS.z_dim):
                    kl_each_dim[i].append(kl_dim[i])                # 将各个维度的kl散度值加入到对应的列表中
                break
            except Exception as e:
                raise e

    with open("./kl_dim_0307.txt", 'w') as f:
        for i in range(FLAGS.z_dim):
            for j in range(FLAGS.epoch):
                f.write(str(kl_each_dim[i][j])+'\t')
            f.write('\n')

    num = 0
    color = []
    for name, hex in matplotlib.colors.cnames.items():
        color.append(str(hex))
        num = num + 1
        if num == 40:
            break

    # -----------------绘制kl值最大的5个特征---------------------
    max_kl = []
    for i in range(FLAGS.z_dim):
        max_kl.append(kl_each_dim[i][FLAGS.epoch - 1])
    index = map(max_kl.index, heapq.nlargest(5, max_kl))
    index = list(index)
    epoch = [i for i in range(1, FLAGS.epoch + 1)]
    _, ax_1 = plt.subplots()
    for i in index:
        ax_1.plot(epoch, kl_each_dim[i], label="latent dim " + str(i), color=color[i])
    ax_1.set_xlabel('epoch')
    ax_1.set_ylabel('KL_loss')
    ax_1.legend()
    plt.savefig("./five_max_figure.jpg")
    plt.show()

    # ----------------绘制全部的40个特征-------------------------
    _, ax_2 = plt.subplots()
    index = map(max_kl.index, heapq.nlargest(40, max_kl))
    index = list(index)
    for i in index:
        ax_2.plot(epoch, kl_each_dim[i], label="latent dim " + str(i), color=color[i])
    ax_2.set_xlabel('epoch')
    ax_2.set_ylabel('KL_loss')
    ax_2.legend()
    plt.savefig("./all_figure.jpg")
    plt.show()


    training_end_time = time.time()
    print("The processing time of program is : {:.2f}mins".format((training_end_time - training_start_time) / 60.0))


if __name__ == '__main__':
    tf.app.run()

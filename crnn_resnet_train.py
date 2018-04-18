import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import numpy as np
import crnn_inference
import crnn_resnet_inference
import sys
from tensorflow.examples.tutorials.mnist import input_data
import win_unicode_console
win_unicode_console.enable()

MODEL_SAVE_PATH = sys.path[0] + '/model/'
KITTI_MODEL_NAME = "KITTI_model.ckpt"
TUM_MODEL_NAME = "TUM_model.ckpt"
SUMMARY_DIR = sys.path[0] + '/log'

KITTI_IMG_PATH = sys.path[0] + '/img_data/128x128/train/'
KITTI_POSE_PATH = sys.path[0] + '/pose_data/train/resize/'
print(KITTI_POSE_PATH)
#TUM_IMG_PATH = 'D:/work/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360/128x128_batchs.h5'
#TUM_POSE_PATH = 'D:/work/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360/gt_sample.h5'

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 1.0
DECAY_STEPS = 1500

def next_batch(X, Y, num):
    return X[num % X.shape[0]], Y[num % Y.shape[0]]

def get_batch(num):
    f = h5py.File(KITTI_IMG_PATH + '/batchs2/' + str(num % 150) + '_img_batch.h5','r')
    img = f['data'][:]
    f.close()

    f = h5py.File(KITTI_POSE_PATH + '/batchs2/' + str(num % 150) + '_pose_batch.h5','r')
    pose = f['data'][:]
    f.close()

    return img, pose

def train(dataset,batch_size):
    x = tf.placeholder("float",[None,128*128*2])
    y = tf.placeholder('float',[None,12])
    x_image = tf.reshape(x, [-1,128,128,2])
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y_conv = crnn_resnet_inference.resnet_inference(x_image, True)
    y_pre = crnn_resnet_inference.rnn_inference(y_conv, batch_size, True, regularizer, dataset)

    global_step = tf.Variable(0, trainable = False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    """
    trans_pre = y_pre[:,0:3].reshape(-1,3)
    rot_pre = y_pre[:,3:].reshape(-1,9)
    trans = y[:,0:3].reshape(-1,3)
    rot = y[:,3:].reshape(-1,9)
    """
    #损失和评价函数
    beta = 100
    cross_entropy_mean = tf.reduce_mean((y_pre - y)**2)
    tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, DECAY_STEPS, LEARNING_RATE_DECAY)
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    merged = tf.summary.merge_all()
    #初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #print('Get pre-trained model......')
        #ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #sess.run(global_step.initializer)
        
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()
        loss_ = []
        for i in range(TRAINING_STEPS):
            X_mb, Y_mb = get_batch(i)
            summary, _, loss_value, step = sess.run([merged, train_op, loss, global_step], feed_dict={x:X_mb, y:Y_mb})
            loss_.append(loss_value)
            summary_writer.add_summary(summary, step)

            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g " % (step, loss_value))
                print('Learning rata: ', learning_rate.eval())
        saver.save(sess, MODEL_SAVE_PATH+KITTI_MODEL_NAME)
        summary_writer.close()
        plt.plot(loss_)
        plt.show()

def main(argv=None):
    print('Geting image datasset and pose dataset...')
    
    dataset = 'KITTI'
    batch_size = 100
    train(dataset, batch_size)
    
if __name__ == '__main__':
    tf.app.run()
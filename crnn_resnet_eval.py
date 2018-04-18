import tensorflow as tf
from PIL import Image
from pylab import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import win_unicode_console
win_unicode_console.enable()
import crnn_resnet_inference
import crnn_resnet_train
import plot_trajectory

KITTI_IMG_PATH = crnn_resnet_train.KITTI_IMG_PATH
KITTI_POSE_PATH = crnn_resnet_train.KITTI_POSE_PATH
TUM_IMG_PATH = 'D:/work/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360/128x128_batchs.h5'
TUM_POSE_PATH = 'D:/work/dataset/TUM/rgbd_dataset_freiburg2_pioneer_360/gt_sample.h5'
number_of_data = 110

def get_batch(num):
    f = h5py.File(KITTI_IMG_PATH + '/batchs/' + str(num % 110 + 454) + '_img_batch.h5','r')
    img = f['data'][:]
    f.close()

    return img

def evaluate(batch_size, dataset):
    out_list = []
    with tf.Graph().as_default() as g:
        x = tf.placeholder("float",[None,128*128*2])
        x_image = tf.reshape(x, [-1,128,128,2])

        y_conv = crnn_resnet_inference.resnet_inference(x_image, False)
        y_pre = crnn_resnet_inference.rnn_inference(y_conv, batch_size, False, None, dataset)

        variable_averages = tf.train.ExponentialMovingAverage(crnn_resnet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(crnn_resnet_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for i in range(number_of_data):
                    X = get_batch(i)
                    y_ = sess.run(y_pre, feed_dict={x:X})
                    out_list.append(y_)
                return out_list
            else:
                print('No checkpoing file found')
                return
            
def main(argv=None):
    print('Geting image datasset and pose dataset...')

    dataset = 'KITTI'        #选择数据集
    
    if dataset == 'KITTI':
        """
        f1 = h5py.File(KITTI_IMG_PATH + '/3_img0_128x512_batchs.h5','r')
        X = f1['data'][:]
        f1.close()
        """
        f2 = h5py.File(KITTI_POSE_PATH + '/1_pose_batchs.h5','r')
        Y = f2['data'][:]
        f2.close()

        batch_size = 10
        #X_mb = X.reshape(-1,batch_size,X.shape[1])
        #Y_mb = Y.reshape(-1,batch_size,Y.shape[1])
        #print(X_mb.shape, Y_mb.shape)
        #Image.fromarray(uint8(X_mb[0,0,:].reshape(128,512,2)[:,:,0]*255)).show()
        out_list = evaluate(batch_size, dataset)
        print(len(out_list))
        out = out_list[0]
        trans = out[:,0:3].reshape(-1,3)
        print(trans.shape)
        out_list.remove(out_list[0])
        for element in out_list:
            trans_ = element[:,0:3].reshape(-1,3)
            trans = np.concatenate([trans,trans_],axis=0)
        print(trans.shape)
        pose = trans
        origin_of_coordinate = np.zeros([1,3])   ###将坐标原点加入到数据中
        pose = np.concatenate((origin_of_coordinate, pose))
        print(pose.shape)
        print('Plot pose......')
        
        #print(pose.shape)
        plot_trajectory.plot_2D_trajectory(pose, Y)  ###得到预测的二维平面轨迹、及其散点图；groundtruth的平面轨迹、散点图
    elif dataset == 'TUM':
        f1 = h5py.File(TUM_IMG_PATH,'r')
        X = f1['data'][:]
        f1.close()

        f2 = h5py.File(TUM_POSE_PATH,'r')
        Y = f2['data'][:]
        f2.close()
        batch_size = 10

        X_mb = X.reshape(-1,batch_size,X.shape[1])
        Y_mb = Y.reshape(-1,batch_size,Y.shape[1])
        out_list = evaluate(X_mb, Y_mb, dataset)
        pose = np.array(out_list).reshape(-1,7)
        gt = Y_mb.reshape(-1,7)
        print(pose.shape)
        
        print('Plot pose......')
        plt.plot(pose[:,0], pose[:,1])
        plt.plot(gt[:,0], gt[:,1], 'r')
        plt.show()

if __name__ == '__main__':
    tf.app.run()
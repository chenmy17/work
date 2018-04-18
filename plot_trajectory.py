import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    figure = plt.figure()
    #ax = figure.add_subplot(111,projection='3d')

    path_kitti = 'D:/work/dataset/KITTI/poses/resize/'
    f = h5py.File(path_kitti+'0_pose_batchs.h5','r')
    pose = f['data'][0:1000]
    f.close()
    
    print(pose.shape)
    print(pose.shape)
    x0 = pose[:,0]
    y0 = pose[:,1]
    z0 = pose[:,2]
    
    plt.plot(x0,y0)
    plt.show()

    #plt.scatter(x,y)
    #plt.show()

    plt.scatter(x0,y0)
    plt.show()

def plot_3D_trajectory(trajectory):
    figure = plt.figure()
    ax = figure.add_subplot(111,projection='3d')

    x = trajectory[:,0]
    y = trajectory[:,1]
    z = trajectory[:,2]

    ax.plot(x,y,z)
    plt.show()

def plot_2D_trajectory(trajectory, gt):
    """绘制给定预测KITTI数据的轨迹，以及真实轨迹"""
    x = trajectory[:,0]
    y = trajectory[:,1]

    #path = 'D:/work/rgbd_dataset_freiburg2_pioneer_slam'
    pose = gt

    x0 = pose[:,0]
    y0 = pose[:,1]
    z0 = pose[:,2]
    plt.plot(x,y)
    plt.plot(x0,y0,'r')
    plt.show()

    plt.scatter(x,y)
    plt.scatter(x0,y0)
    plt.show()
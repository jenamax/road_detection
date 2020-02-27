from cyber_py3 import cyber
from modules.drivers.proto.pointcloud_pb2 import PointCloud
import numpy as np
from math import factorial, log
from numpy.random import uniform
from scipy.linalg import null_space
from scipy.special import comb
from modules.common.proto.header_pb2 import Header
import time
import copy
from numpy import std, transpose


def plane_3points(p1, p2, p3):  # compute plane parameters
    return null_space(np.array([[p1.x, p1.y, p1.z, 1], [p2.x, p2.y, p2.z, 1], [p3.x, p3.y, p3.z, 1]]))

def ransac_plane(data):
    points = data.point
    mss = 3
    w = 0.5
    trsh = 0.2
    A, B, C, D = 0, 0, 0, 0
    inliers = []
    prev_inliers = []
    msg = copy.deepcopy(data)

    Z = np.array([p.z for p in points])
    sigma_trsh = 0.7
    mean_z = sum(Z) / len(Z)
    std_z = std(Z)
    k = 3
    del msg.point[:]
    msg.point.extend(
        ([p for p in data.point if abs(p.z - mean_z) < sigma_trsh * std_z and 12.5 > p.x > 0 and abs(p.y) < 7]))  # crop ROI
    XYZ = np.array([[p.x, p.y, p.z, 1] for p in msg.point])
    print("ROI point num: ", len(msg.point))
    for i in range(0, k):
        p1 = int(uniform(0, len(msg.point)))
        p2 = int(uniform(0, len(msg.point)))
        while p2 == p1:
            p2 = int(uniform(0, len(msg.point)))
        p3 = int(uniform(0, len(msg.point)))
        while p3 == p1 or p3 == p2:
            p3 = int(uniform(0, len(msg.point)))

        a, b, c, d = plane_3points(points[p1], points[p2], points[p3])
        new_msg = copy.deepcopy(data)
        del new_msg.point[:]
        paprams = np.array([a, b, c, d], dtype=np.float32)
        dist = XYZ.dot(paprams) / (a ** 2 + b ** 2 + c ** 2) ** .5 # compute distances from each point to plane
        # dist = np.array([(a * p.x + b * p.y + c * p.z + d) / (a ** 2 + b ** 2 + c ** 2) ** .5 for p in msg.point])
        # inliers = np.array([msg.point[i] for i in range(0, len(msg.point)) if dist[i] < trsh])
        inliers = np.array(
            [msg.point[i] for i in range(len(msg.point)) if dist[i] < trsh]) # write inliers to array
        # print("Inliers num: ", len(inliers))
        if len(prev_inliers) < len(inliers): # select best plane
            A, B, C, D = a, b, c, d
            new_msg.point.extend(inliers)
            msg = new_msg
        prev_inliers = np.copy(inliers)
    #     print("Inliers percent: ", float(len(inliers)) / len(points))
    # print("road points plane: ", len(msg.point))
    return A, B, C, D, msg


global count, t_mean
count = 0
t_mean = 0


def callback(data):
    """
    Reader message callback.
    """
    global count
    global t_mean
    s = time.time()
    points = data.point
    msg = copy.deepcopy(data)
    try:
        msg = ransac_plane(data)[4] # create point cloud with road points

        road_writer = lidar_detect_node.create_writer("/road_points", PointCloud, 6)
        dt = time.time() - s
        print(dt)
        count += 1
        t_mean = t_mean * (count - 1) / count + dt / count
        print(t_mean)
        print()
        road_writer.write(msg)

    except KeyboardInterrupt:
        cyber.shutdown()
        exit(0)


def lidar_listener():
    """
    Reader message.
    """
    print("LIDAR detection started")
    lidar_detect_node.create_reader("/apollo/sensor/lidar128/compensator/PointCloud2", PointCloud, callback) # subscribe to LIDAR messages
    lidar_detect_node.spin()


if __name__ == '__main__':
    cyber.init()
    lidar_detect_node = cyber.Node("lidar_road_detect")
    lidar_listener()
    cyber.shutdown()

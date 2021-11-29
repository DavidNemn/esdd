#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rospkg
import numpy as np
from rospy.numpy_msg import numpy_msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import String
from tf import TransformListener
from tf.transformations import quaternion_matrix
import cv2
import time
import yaml

# TriggerMapExpansion类, 发布扩展全局地图的命令
# 线程状态
DISABLED = 'DISABLED'
WAIT_FOR_MAP = 'WAIT_FOR_MAP'
CHECKING = 'CHECKING'

class TriggerMapExpansion:
    _state = WAIT_FOR_MAP
    _map = None
    _t_map = None
    _got_camera_info = False

    def __init__(self):
        self._rate = rospy.get_param('~rate', 3)                                    # 发布频率: 3Hz
        self._visibility_th = rospy.get_param('~visibility_threshold', .75)         # 可视化阈值
        self._coverage_th = rospy.get_param('~coverage_threshold', .3)              # 收敛阈值
        self._baseline_th = rospy.get_param('~baseline_threshold', .1)              # 基线阈值
        self._dvs_frame_id = rospy.get_param('dvs_frame_id', 'dvs_evo')             # 相机坐标系id
        self._world_frame_id = rospy.get_param('world_frame_id', 'world')           # 世界坐标系id
        self._map_to_skip = rospy.get_param('number_of_initial_maps_to_skip', 0)    # 跳过初始地图个数

        with open(rospy.get_param('calib_file', ''), 'r') as stream:
            try:
                cam_info = yaml.load(stream, yaml.SafeLoader)
                self._w = cam_info['image_width']
                self._h = cam_info['image_height']
                self._K = np.array(cam_info['camera_matrix']['data']).reshape((3, 3))
            except yaml.YAMLError as exc:
                print(exc)

        self._remote = rospy.Publisher('remote_key', String, queue_size=1)  # 发布远程命令
        self._tf = TransformListener(True)                                  # 位姿变换

        rospy.Subscriber("pointcloud", PointCloud2, self._MapCallback)          # 订阅点云
        rospy.Subscriber("remote_key", String, self._RemoteKeyCallback)         # 订阅远程命令
        rospy.Subscriber("camera_info", CameraInfo, self._CameraInfoCallback)   # 订阅相机信息
        rospy.Timer(rospy.Duration(1./self._rate), self._CheckNewMapNeeded)     # 以rate的频率检查是否需要新的局部地图

        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)

    def _CameraInfoCallback(self, msg):
        if self._got_camera_info:
            return

        self._w = msg.width
        self._h = msg.height
        self._K = np.array(msg.K).reshape((3, 3))

        self._got_camera_info = True

    def _RemoteKeyCallback(self, msg):
        m = msg.data

        if (m == 'disable_map_expansion'):
            self._state = DISABLED
        if (m == 'enable_map_expansion'):
            self._state = CHECKING

    def _MapCallback(self, msg):
        if self._state == DISABLED:
            return

        map = []
        for p in pc2.read_points(msg):
            map.append([p[0], p[1], p[2], 1.])

        self._map = np.array(map).T
        # rospy.loginfo('Received map: {} points'.format(len(map)))   # 收到map

        try:
            now = rospy.Time(0)
            self._tf.waitForTransform(self._dvs_frame_id, self._world_frame_id, now, rospy.Duration(1.))
            (self._t_map, q) = self._tf.lookupTransform(self._dvs_frame_id, self._world_frame_id, now)
            self._state = CHECKING
        except:
            self._t_map = None

    def _CheckNewMapNeeded(self, event):
        if self._state != CHECKING:
            return

        if (self._t_map and self._map.size == 0):
            return

        try:
            now = rospy.Time(0)
            self._tf.waitForTransform(self._dvs_frame_id, self._world_frame_id, now, rospy.Duration(1.))
        except:
            return

        (t, q) = self._tf.lookupTransform(self._dvs_frame_id, self._world_frame_id, now)

        T = quaternion_matrix(q)
        T[:3, 3] = t
        
        pts = T.dot(self._map)[:3, ]    # 将点云投影到相机坐标系

        coverage, visibility = self._MapVisibility(pts)     # 获取地图可视程度

        BoD = self._BaselineOverDepth(pts, np.subtract(t, self._t_map)) if self._t_map else 0.  # 计算基线/深度

        if coverage < self._coverage_th or visibility < self._visibility_th or BoD > self._baseline_th:
            # 发布地图更新命令
            # rospy.loginfo('Sending update, coverage: {} %, map visibility: {} %, baseline/depth: {}'.format(coverage * 100, visibility * 100, BoD))
            self._state = WAIT_FOR_MAP
            self._remote.publish('update')

    def _MapVisibility(self, pts):
        pts = pts[:, pts[2, ] > 0]
        pts = self._K.dot(pts/pts[2, ]).T

        N = len(pts)

        if N == 0:
            return 0.

        mask = np.zeros((self._h, self._w), np.uint8)
        cnt = 0.
        for i in xrange(N):
            x, y = pts[i][:2]
            cv2.circle(mask, (int(x), int(y)), 7, 255, -1)
            if x >= 0 and y >= 0 and x < self._w and y < self._h:
                cnt += 1.

        return float(cv2.countNonZero(mask))/(self._w*self._h), float(cnt) / N

    def _BaselineOverDepth(self, pts, t):
        avg_depth = np.average(np.linalg.norm(pts, axis=0))
        baseline = np.linalg.norm(t)

        return baseline / avg_depth

if __name__ == '__main__':
    rospy.init_node('trigger_map_expansion')
    node = TriggerMapExpansion()
    rospy.spin()

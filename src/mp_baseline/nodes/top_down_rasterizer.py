#!/usr/bin/python
# coding=utf-8

from __future__ import print_function
from collections import defaultdict
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import DetectedObjectArray
from tf.transformations import euler_from_quaternion
from PIL import Image
import numpy as np
import rospy
import math
import cv2
import os


class TopDownRasterizer:

    def __init__(self):

        self.n_track_history = defaultdict(list)
        self.curr_tracks = []
        self.curr_ego_pose = None

        # lane_map_path = rospy.get_param("map_path")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lane_map = cv2.imread(os.path.join(dir_path, 'demo_route_large.png'), 0)

        # Binarize lane map
        self.lane_map[self.lane_map == -1] = 0
        self.lane_map[self.lane_map == 1] = 255

        self.map_tl_x = rospy.get_param("map_tl_x", default=8926)
        self.map_tl_y = rospy.get_param("map_tl_y", default=10289)

        # Subscribers
        rospy.Subscriber("cvkf_tracked_objects", DetectedObjectArray, self.detections_with_tracks)
        rospy.Subscriber("current_pose", PoseStamped, self.curr_pose_callback)

    def detections_with_tracks(self, msg):
        """ Callback to get detected objects

        Parameters
        ----------
        msg : DetectedObjectArray
        """

        self.curr_tracks = []

        for detection_object in msg.objects:

            if detection_object.id in self.n_track_history:
                if len(self.n_track_history[detection_object.id]) == 10:
                    del self.n_track_history[detection_object.id][0]

            self.n_track_history[detection_object.id].append({
                "pose": detection_object.pose,
                "dimensions": detection_object.dimensions,
                "velocity": detection_object.velocity,
                "acceleration": detection_object.acceleration
            })

            self.curr_tracks.append(detection_object.id)

        # Remove old tracks
        for key in list(self.n_track_history):
            if key not in self.curr_tracks:
                del self.n_track_history[key]

        self.process_n_tracks()

    def curr_pose_callback(self, msg):
        """ Callback to get current pose of ego vehicle

        Parameters
        ----------
        msg : PoseStamped
        """
        self.curr_ego_pose = msg

    def process_n_tracks(self):
        """ Process tracked objects """

        if self.curr_ego_pose is None:
            return
        keys_list = list(self.n_track_history.keys())

        if len(keys_list) > 0:
            points = []
            rotations = []
            dimensions = []

            for list_idx in range(len(keys_list)):

                if (len(self.n_track_history[keys_list[list_idx]]) > 1):
                    curr_pose = self.n_track_history[keys_list[list_idx]][-1]["pose"]

                    for i in range(0, len(self.n_track_history[keys_list[list_idx]])):
                        points.append([
                            curr_pose.position.x, 
                            curr_pose.position.y,
                            curr_pose.position.z
                        ])
                        rotations.append([
                            curr_pose.orientation.x, 
                            curr_pose.orientation.y,
                            curr_pose.orientation.z, 
                            curr_pose.orientation.w
                        ])
                        dimensions.append([
                            self.n_track_history[keys_list[list_idx]][i]["dimensions"].x, 
                            self.n_track_history[keys_list[list_idx]][i]["dimensions"].y,
                            self.n_track_history[keys_list[list_idx]][i]["dimensions"].z
                        ])

            # Add ego vehicle too for testing
            points.append([0, 0, 0])
            rotations.append([0, 0, 0, 1])
            dimensions.append([5.5, 2.0, 1.5])

            if len(points) > 1:
                self.rasterize(obstacle_points=self.get_associated_polygon_points(points, rotations, dimensions))

    def get_associated_polygon_points(self, points, rotations, dimensions):
        """ Create 4 polygon points for convex hull.

        Parameters
        ----------
        points : list of 
            List of centroids of detected objects
        rotations : float
            List of quaterinion rotations of detected objects
        dimensions : list
            List of 3D dimensions of detected objects

        Returns
        -------
        np.ndarray
            A numpy array of shape (5, n, 3) where row one corresponds to centroid points and row 1-4 are lists of corresponding polygon points
        """

        centroids_and_poly_points = [[], [], [], [], []]

        for point, rotation, dimension in zip(points, rotations, dimensions):

            # Halved dimensions for creating polygon points
            halved_dim = np.array(dimension) / 2

            # Get rotation in yaw as rotation matrix
            yaw = euler_from_quaternion(
                [rotation[0], rotation[1], rotation[2], rotation[3]])[2]
            rotation_matrix = self.euler_to_rotMat(0, 0, yaw)

            # Rotate polygon points around origin
            rotated_vec_0 = rotation_matrix.dot(
                np.array([halved_dim[0], halved_dim[1], halved_dim[2]]))
            rotated_vec_1 = rotation_matrix.dot(
                np.array([-halved_dim[0], halved_dim[1], halved_dim[2]]))
            rotated_vec_2 = rotation_matrix.dot(
                np.array([-halved_dim[0], -halved_dim[1], halved_dim[2]]))
            rotated_vec_3 = rotation_matrix.dot(
                np.array([halved_dim[0], -halved_dim[1], halved_dim[2]]))

            centroids_and_poly_points[0].append(point)
            centroids_and_poly_points[1].append([point[0] + rotated_vec_0[0], point[1] + rotated_vec_0[1], point[2]])
            centroids_and_poly_points[2].append([point[0] + rotated_vec_1[0], point[1] + rotated_vec_1[1], point[2]])
            centroids_and_poly_points[3].append([point[0] + rotated_vec_2[0], point[1] + rotated_vec_2[1], point[2]])
            centroids_and_poly_points[4].append([point[0] + rotated_vec_3[0], point[1] + rotated_vec_3[1], point[2]])

        return np.array(centroids_and_poly_points)

    def euler_to_rotMat(self, roll, pitch, yaw):
        """ Get rotation matrix from euler angles

        Parameters
        ----------
        roll : float
            Roll in radians
        pitch : float
            Pitch in radians
        yaw : float
            Yaw in radians

        Returns
        -------
        np.ndarray
            A 3x3 numpy array having the rotation matrix
        """

        Rz_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Ry_pitch = np.array([[np.cos(pitch), 0,
                              np.sin(pitch)], [0, 1, 0],
                             [-np.sin(pitch), 0,
                              np.cos(pitch)]])
        Rx_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

        return np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))

    def rasterize(self, obstacle_points=None, side_range=(-100, 100), fwd_range=(-100, 100), res=0.2):
        """ Rasterize obstacles and vector map

        Parameters
        ----------
        obstacle_points : np.ndarray
            A numpy array of shape (5, n, 3) where row one corresponds to centroid points and row 1-4 are lists of corresponding polygon points
        side_range : tuple
            Horizontal range in meters for cropping
        fwd_range : tuple
            Vertical range in meters for cropping
        res : float
            Meters per pixel resolution
        """
        
        # Filter obstacles outside side & forward range
        x_filtered = np.logical_and((obstacle_points[0][:, 0] > fwd_range[0]),
                                    (obstacle_points[0][:, 0] < fwd_range[1]))
        y_filtered = np.logical_and((obstacle_points[0][:, 1] > -side_range[1]),
                                    (obstacle_points[0][:, 1] < -side_range[0]))
        # TODO apply filter maybe later
        obstacle_filter = np.argwhere(np.logical_and(x_filtered, y_filtered)).flatten()

        # 3D map frame to top-down image transform matrix
        # Here x = -y, y=-x, and coloumn 2 is use to translate points to shift image origin
        trans_3d_to_img = np.array([
            [0, -1 / res, -math.floor(side_range[0] / res)],
            [-1 / res, 0, -math.floor(fwd_range[0] / res)], 
            [0, 0, 1]
        ])

        obstacle_points_img = [[], [], [], [], []]
        for idx, obstacle_point_arr in enumerate(obstacle_points):
            
            # Transform 3D centroids and polygon points to top-down image
            points_img = np.concatenate([obstacle_point_arr[:, :2], np.ones((obstacle_point_arr[:, :2].shape[0], 1))], axis=1)
            points_img = np.dot(points_img, trans_3d_to_img.T).astype(np.int32)[:, :2]

            obstacle_points_img[idx] = points_img
        
        # Get shape of image with
        x_max = int((side_range[1] - side_range[0]) / res)
        y_max = int((fwd_range[1] - fwd_range[0]) / res)

        # Create numpy array of zeros with above acquired shape
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        # Draw min rect for each obstacle
        im = self.draw_min_rects(im, obstacle_points_img[1:])
        # Merge occupancy map and vector map raster
        im = self.merge_vector_map(im)
        # Convert from numpy array to a PIL image
        im = Image.fromarray(im)


    def merge_vector_map(self, occupancy_img):

        quaternion = (self.curr_ego_pose.pose.orientation.x,
                      self.curr_ego_pose.pose.orientation.y,
                      self.curr_ego_pose.pose.orientation.z,
                      self.curr_ego_pose.pose.orientation.w)
        
        yaw_rotation = math.degrees(euler_from_quaternion(quaternion)[2])

        ego_index = [
            int(math.ceil(abs(self.map_tl_x - self.curr_ego_pose.pose.position.x) / 0.2) - 2),
            int(math.ceil(abs(self.map_tl_y - self.curr_ego_pose.pose.position.y) / 0.2) + 2)
        ]

        yaw_rotation -= 88

        cropped_map = self.lane_map[ego_index[1] - 500:ego_index[1] + 500,
                                    ego_index[0] - 500:ego_index[0] + 500]
        cropped_map = self.rotate_image(cropped_map, -yaw_rotation)

        merged_img = cv2.add(cropped_map, occupancy_img)

        cv2.imshow('image', merged_img)
        cv2.waitKey(1)

        return merged_img

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
        return result

    def draw_min_rects(self, img, polygon_points_img):
        for idx, _ in enumerate(polygon_points_img[0]):
            poly_point_0 = polygon_points_img[0][idx, :2]
            poly_point_1 = polygon_points_img[1][idx, :2]
            poly_point_2 = polygon_points_img[2][idx, :2]
            poly_point_3 = polygon_points_img[3][idx, :2]
            cnt = np.array([[poly_point_0], [poly_point_1], [poly_point_2],
                            [poly_point_3], [poly_point_0]])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 255, 255), -1)
        return img

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('top_down_rasterizer',
                    anonymous=False,
                    log_level=rospy.INFO)
    node = TopDownRasterizer()
    node.run()

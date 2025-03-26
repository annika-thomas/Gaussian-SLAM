import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import pandas as pd
from glob import glob
from scipy.spatial.transform import Rotation as R
import copy
import bisect
from typing import List, Tuple, Set
import numpy as np

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import copy

def interpolate_pose_gaps_odom(pose_data, max_normal_gap=0.2, target_frequency=10):
    """
    Interpolate gaps in pose data that exceed max_normal_gap seconds.
    """
    # Convert timestamps to seconds if they're in nanoseconds
    df = copy.deepcopy(pose_data)
    if df['timestamp'].max() > 1e10:  # Assuming it's in nanoseconds
        df['timestamp'] = df['timestamp'] / 1e9
    
    # Sort by timestamp and reset index
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Create new DataFrame with interpolated data
    new_rows = []
    
    # Iterate through consecutive rows
    for i in range(len(df) - 1):
        start_time = df.iloc[i]['timestamp']
        end_time = df.iloc[i + 1]['timestamp']
        gap_duration = end_time - start_time
        
        # Only interpolate if gap is too large
        if gap_duration > max_normal_gap:
            # Calculate number of points to interpolate
            n_points = max(1, int(gap_duration * target_frequency))
            
            # print(f"Gap from {start_time:.3f} to {end_time:.3f} ({gap_duration:.3f}s): inserting {n_points} points")
            
            # Create new timestamps
            new_timestamps = np.linspace(start_time, end_time, n_points + 2)[1:-1]
            
            # Get start and end positions
            start_pos = df.iloc[i][['x', 'y', 'z']].values
            end_pos = df.iloc[i + 1][['x', 'y', 'z']].values
            
            # Linear interpolation for position
            alphas = np.linspace(0, 1, n_points + 2)[1:-1]
            interpolated_positions = np.array([start_pos * (1-alpha) + end_pos * alpha for alpha in alphas])
            
            # Quaternion interpolation using SLERP
            start_quat = df.iloc[i][['qw', 'qx', 'qy', 'qz']].values
            end_quat = df.iloc[i + 1][['qw', 'qx', 'qy', 'qz']].values
            
            # Create rotation objects for SLERP
            key_rots = Rotation.from_quat(np.vstack([start_quat[[1,2,3,0]], end_quat[[1,2,3,0]]]))
            key_times = np.array([start_time, end_time])
            slerp = Slerp(key_times, key_rots)
            
            # Interpolate orientations
            interpolated_rots = slerp(new_timestamps)
            interpolated_quats = interpolated_rots.as_quat()
            # Convert from scipy convention (x,y,z,w) to original convention (w,x,y,z)
            interpolated_quats = np.roll(interpolated_quats, 1, axis=1)
            
            # Create new rows for the interpolated points
            for j in range(len(new_timestamps)):
                new_rows.append({
                    'timestamp': new_timestamps[j],
                    'x': interpolated_positions[j,0],
                    'y': interpolated_positions[j,1],
                    'z': interpolated_positions[j,2],
                    'qw': interpolated_quats[j,0],
                    'qx': interpolated_quats[j,1],
                    'qy': interpolated_quats[j,2],
                    'qz': interpolated_quats[j,3]
                })
    
    # Add interpolated rows to original DataFrame
    if new_rows:
        interpolated_df = pd.DataFrame(new_rows)
        df = pd.concat([df, interpolated_df], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def interpolate_pose_gaps(pose_data, max_normal_gap=0.1, target_frequency=100):
    """
    Interpolate gaps in pose data that exceed max_normal_gap seconds.
    
    Args:
        pose_data: DataFrame with columns ['#timestamp_kf', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        max_normal_gap: Maximum allowed gap (in seconds) before interpolation (default: 0.1s)
        target_frequency: Target frequency for interpolation points (Hz) (default: 100Hz)
    
    Returns:
        DataFrame with interpolated pose data
    """
    # Convert timestamps to seconds if they're in nanoseconds
    df = pose_data.copy()
    if df['#timestamp_kf'].max() > 1e10:  # Assuming it's in nanoseconds
        df['#timestamp_kf'] = df['#timestamp_kf'] / 1e9
    
    # Sort by timestamp to ensure proper interpolation
    df = df.sort_values('#timestamp_kf')
    
    # Find gaps larger than max_normal_gap
    time_diffs = df['#timestamp_kf'].diff()
    large_gaps = time_diffs[time_diffs > max_normal_gap]
    
    if large_gaps.empty:
        return df
    
    # Create new DataFrame with interpolated data
    new_rows = []
    
    for idx in large_gaps.index:
        # Get start and end points of the gap
        start_time = df.loc[idx-1, '#timestamp_kf']
        end_time = df.loc[idx, '#timestamp_kf']
        gap_duration = end_time - start_time
        
        # Calculate number of points to interpolate
        n_points = int(gap_duration * target_frequency)
        
        # Create new timestamps
        new_timestamps = np.linspace(start_time, end_time, n_points + 2)[1:-1]
        
        # Get start and end positions
        start_pos = df.loc[idx-1, ['x', 'y', 'z']].values
        end_pos = df.loc[idx, ['x', 'y', 'z']].values
        
        # Linear interpolation for position
        alphas = np.linspace(0, 1, n_points + 2)[1:-1]
        interpolated_positions = np.array([start_pos * (1-alpha) + end_pos * alpha for alpha in alphas])
        
        # Quaternion interpolation using SLERP
        start_quat = df.loc[idx-1, ['qw', 'qx', 'qy', 'qz']].values
        end_quat = df.loc[idx, ['qw', 'qx', 'qy', 'qz']].values
        
        # Create rotation objects for SLERP
        key_rots = Rotation.from_quat(np.vstack([start_quat[[1,2,3,0]], end_quat[[1,2,3,0]]]))
        key_times = np.array([start_time, end_time])
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate orientations
        interpolated_rots = slerp(new_timestamps)
        interpolated_quats = interpolated_rots.as_quat()
        # Convert from scipy convention (x,y,z,w) to original convention (w,x,y,z)
        interpolated_quats = np.roll(interpolated_quats, 1, axis=1)
        
        # Create new rows for the interpolated points
        for i in range(len(new_timestamps)):
            new_rows.append({
                '#timestamp_kf': new_timestamps[i],
                'x': interpolated_positions[i,0],
                'y': interpolated_positions[i,1],
                'z': interpolated_positions[i,2],
                'qw': interpolated_quats[i,0],
                'qx': interpolated_quats[i,1],
                'qy': interpolated_quats[i,2],
                'qz': interpolated_quats[i,3]
            })
    
    # Add interpolated rows to original DataFrame
    if new_rows:
        interpolated_df = pd.DataFrame(new_rows)
        df = pd.concat([df, interpolated_df], ignore_index=True)
        df = df.sort_values('#timestamp_kf').reset_index(drop=True)
    
    return df


# def transform_coordinate_system(poses):
#     """
#     Transform poses from:
#         X forward, Y left, Z up
#     to:
#         X right, Y down, Z forward
#     Only transforms the rotation part, keeping positions unchanged.
#     """
#     # R = np.array([
#     #     [0, -1,  0],
#     #     [0,  0, -1],
#     #     [1,  0,  0]
#         # ])

#     # First transformation (original to camera frame)
#     R_first = np.array([
#         [0, -1,  0],
#         [0,  0, -1],
#         [1,  0,  0]
#     ])

#     # Create a Rotation object for the first transformation
#     rot_first = R.from_matrix(R_first)

#     # 180-degree rotation about the Y-axis (in the camera frame)
#     rot_y_180 = R.from_euler('z', 180, degrees=True)

#     # Combine the two rotations
#     rot_final = rot_y_180 * rot_first

#     # Get the final rotation matrix
#     R_final = rot_final.as_matrix()

#     print("Final transformation matrix:")
#     print(R_final)

#     # Combined rotation: Apply z-axis rotation after x-axis rotation


#     rotated_poses = []
#     for pose in poses:
#         if pose.shape == (4, 4):  # Transformation matrix
#             rotation_part = pose[:3, :3]  # Extract the rotation matrix
#             rotated_rotation = np.dot(R_final, rotation_part)
#             rotated_pose = pose.copy()
#             rotated_pose[:3, :3] = rotated_rotation  # Replace the rotation matrix
#         elif pose.shape == (3, 3):  # Rotation matrix only
#             rotated_pose = np.dot(R_final, pose)
#         else:
#             raise ValueError(f"Invalid pose shape {pose.shape}, expected (4, 4) or (3, 3).")

#         rotated_poses.append(rotated_pose)

#     return rotated_poses


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} RGB frames")
        print(f"Loaded {len(self.depth_paths)} Depth frames")
        print(f"Loaded {len(self.poses)} poses")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        print(f"Looking for color data at {index}")
        print(f"Loaded {len(self.color_paths)} RGB frames")
        print("color data type: ", type(color_data))
        print("color data shape: ", color_data.shape)
        
    
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, None, self.poses[index]


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, None, self.poses[index]
    
    

class Kimera(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses, self.timestamps = self.load_kimera(
            self.dataset_path, frame_rate=-1)

    def load_kimera(self, datapath, frame_rate=-1):
        """Read video data in Kimera format"""
        # Read pose data from CSV
        csv_files = [f for f in os.listdir(datapath) if f.endswith('.csv')]
        print(csv_files)
        if not csv_files:
            raise FileNotFoundError(f"No .csv file found in {datapath}")
        pose_file = next((f for f in csv_files if f.endswith('gt_odom.csv')), None)
        if pose_file is None:
            raise FileNotFoundError(f"No .csv file ending with 'gt_odom.csv' found in {datapath}")
        pose_file = os.path.join(datapath, pose_file)

        odometry_file = next((f for f in csv_files if f.endswith('vio.csv')), None)
        if odometry_file is None:
            raise FileNotFoundError(f"No .csv file ending with 'vio.csv' found in {datapath}")
        odometry_file = os.path.join(datapath, odometry_file)

        # # Read pose data from the .csv file with specific column names
        # pose_data = pd.read_csv(pose_file)
        # odometry_data = pd.read_csv(odometry_file)
        
        # # Extract timestamps and pose data
        # timestamps = pose_data['#timestamp_kf'].values
        # positions = pose_data[['x', 'y', 'z']].values
        # quaternions = pose_data[['qw', 'qx', 'qy', 'qz']].values

        # # Extract timestamps and odometry data
        # odometry_timestamps = odometry_data['timestamp'].values
        # odometry_positions = odometry_data[['x', 'y', 'z']].values
        # odometry_quaternions = odometry_data[['qw', 'qx', 'qy', 'qz']].values

        # # print("positions shape: ", positions.shape)
        # # print("quaternions shape: ", quaternions.shape)

        # # print("odometry positions shape: ", odometry_positions.shape)
        # # print("odometry quaternions shape: ", odometry_quaternions.shape)
        
        # print(f"Loaded {len(timestamps)} poses from {pose_file}")
        # print(f"Loaded {len(odometry_timestamps)} vio poses from {odometry_file}")
        
        # # Get all color and depth image paths
        # color_pattern = os.path.join(datapath, 'color', 'color_*.png')
        # depth_pattern = os.path.join(datapath, 'depth', 'depth_*.png')
        
        # color_paths = sorted(glob(color_pattern))
        # depth_paths = sorted(glob(depth_pattern))

        # print(f"Found {len(color_paths)} color images and {len(depth_paths)} depth images")
        
        # # Extract timestamps from filenames
        # def extract_timestamp(path):
        #     return float(path.split('_')[-1].split('.')[0])
        
        # color_timestamps = np.array([extract_timestamp(p) for p in color_paths])
        # depth_timestamps = np.array([extract_timestamp(p) for p in depth_paths])

        # print("number of color timestamps: ", len(color_timestamps))


        # Read pose data from the .csv file with specific column names
        pose_data = pd.read_csv(pose_file)
        odometry_data = pd.read_csv(odometry_file)

        odometry_data = odometry_data.sort_values('timestamp')
        pose_data = pose_data.sort_values('#timestamp_kf')

        # Extract timestamps and pose data
        timestamps = pose_data['#timestamp_kf'].values
        positions = pose_data[['x', 'y', 'z']].values
        quaternions = pose_data[['qw', 'qx', 'qy', 'qz']].values

        # Extract timestamps and odometry data
        odometry_timestamps = odometry_data['timestamp'].values
        odometry_positions = odometry_data[['x', 'y', 'z']].values
        odometry_quaternions = odometry_data[['qw', 'qx', 'qy', 'qz']].values

        # # convert timestamps to seconds (from ros time)
        timestamps = timestamps / 1e9
        odometry_timestamps = odometry_timestamps / 1e9

        # Interpolate the pose data
        interpolated_pose_data = interpolate_pose_gaps(pose_data, max_normal_gap=0.01, target_frequency=100)

        # Get new timestamps array
        interpolated_timestamps = interpolated_pose_data['#timestamp_kf'].values

        # Interpolate the odometry data

        interpolated_odometry_data = interpolate_pose_gaps_odom(odometry_data, max_normal_gap=0.01, target_frequency=100)
        # interpolated_odometry_data = odometry_data

        # Get new timestamps array
        interpolated_odometry_timestamps = interpolated_odometry_data['timestamp'].values

        # Get all color and depth image paths
        color_pattern = os.path.join(datapath, 'color', 'color_*.png')
        depth_pattern = os.path.join(datapath, 'depth', 'depth_*.png')

        color_paths = sorted(glob(color_pattern))
        depth_paths = sorted(glob(depth_pattern))

        print(f"Found {len(color_paths)} color images and {len(depth_paths)} depth images")

        # Extract timestamps from filenames
        def extract_timestamp(path):
            return float(path.split('_')[-1].split('.')[0])

        color_timestamps = np.array([extract_timestamp(p) for p in color_paths])
        depth_timestamps = np.array([extract_timestamp(p) for p in depth_paths])

        # conver tto seconds 
        color_timestamps = color_timestamps / 1e9
        depth_timestamps = depth_timestamps / 1e9

        # add 0.1 seconds to the odometry timestamps
        interpolated_odometry_timestamps = interpolated_odometry_timestamps + 0.02

        # Associate frames based on timestamps
        associations = self.associate_frames(color_timestamps, depth_timestamps, interpolated_timestamps, interpolated_odometry_timestamps, max_dt=0.01)


        # Extract timestamps and pose data
        interp_timestamps = interpolated_pose_data['#timestamp_kf'].values
        interp_positions = interpolated_pose_data[['x', 'y', 'z']].values
        interp_quaternions = interpolated_pose_data[['qw', 'qx', 'qy', 'qz']].values

        # Extract timestamps and odometry data
        interp_odometry_timestamps = interpolated_odometry_data['timestamp'].values
        interp_odometry_positions = interpolated_odometry_data[['x', 'y', 'z']].values
        interp_odometry_quaternions = interpolated_odometry_data[['qw', 'qx', 'qy', 'qz']].values
        
        # Associate frames
        # associations = self.associate_frames(color_timestamps, depth_timestamps, timestamps, odometry_timestamps, max_dt=20000000)
        
        # Apply frame rate filtering if specified
        if frame_rate > 0:
            indices = [0]
            for i in range(1, len(associations)):
                t0 = color_timestamps[associations[indices[-1]][0]]
                t1 = color_timestamps[associations[i][0]]
                if t1 - t0 > 1.0 / frame_rate:
                    indices += [i]
        else:
            indices = range(len(associations))

        # print(f"Selected {len(indices)} frames for processing")
        
        # Build synchronized lists
        images, depths, poses, odoms, sync_timestamps = [], [], [], [], []
        inv_first_pose = None
        c2w = np.eye(4)
        c2w_odom = np.eye(4)

        print("number of associations: ", len(associations))

        for ix in indices:
            (i, j, k, l) = associations[ix]
            images.append(color_paths[i])
            depths.append(depth_paths[j])
            sync_timestamps.append(interp_timestamps[k])

            # print("association: ", associations[ix])

            # print("positions 1: ", positions[0])
            
            # Get position and quaternion for this frame
            position = interp_positions[k]
            quaternion = interp_quaternions[k]

            # print("position: ", position)
            # print("quaternion: ", quaternion)

            # print("position: ", position)
            
            # Convert to matrix using position and quaternion directly
            c2w = self.pose_matrix_from_position_quaternion(position, quaternion)
            # print("c2w: ", c2w.astype(np.float32))
            
            # rotate pose 90 degrees about x-axis using scipy Rotation on rotation part of c2w
            c2w_new = copy.deepcopy(c2w)
            
            # Extract rotation part
            c2w_R = c2w[:3, :3]
            
            # Convert to Rotation object
            rot = R.from_matrix(c2w_R)
            
            # Create rotations about local axes
            rot_x = R.from_euler('y', 90, degrees=True)
            rot_z = R.from_euler('z', 270, degrees=True)
            
            # Apply rotations in reverse order for local axes
            # The original rotation represents the camera's orientation
            # We multiply on the right for local axes rotations
            new_rot = rot * rot_x * rot_z

            # print("new rot pose shape: ", new_rot.as_matrix().shape)
            
            # Convert back to matrix
            c2w_new[:3, :3] = new_rot.as_matrix()

            c2w = c2w_new

            # print("c2w shape: ", c2w.shape)

            poses.append(c2w.astype(np.float32))

            # print("c2w as type float32: ", c2w.astype(np.float32))

            # Get position and quaternion for this frame
            position_odom = interp_odometry_positions[l]
            quaternion_odom = interp_odometry_quaternions[l]

            # print("position odom: ", position_odom)
            # print("quaternion odom: ", quaternion_odom)

            # print("position: ", position)
            
            # Convert to matrix using position and quaternion directly
            c2w_odom = self.pose_matrix_from_position_quaternion(position_odom, quaternion_odom)
            # print("c2w: ", c2w.astype(np.float32))
            
            # rotate pose 90 degrees about x-axis using scipy Rotation on rotation part of c2w
            c2w_new_odom = copy.deepcopy(c2w_odom)
            
            # Extract rotation part
            c2w_R_odom = c2w_odom[:3, :3]
            
            # Convert to Rotation object
            rot_odom = R.from_matrix(c2w_R_odom)
            
            # Create rotations about local axes
            # rot_x = R.from_euler('y', 90, degrees=True)
            # rot_z = R.from_euler('z', 270, degrees=True)
            
            # Apply rotations in reverse order for local axes
            # The original rotation represents the camera's orientation
            # We multiply on the right for local axes rotations
            # new_rot_odom = rot_odom * rot_x * rot_z

            new_rot_odom = rot_odom

            # print("new rot odom shape: ", new_rot_odom.as_matrix().shape)
            
            # Convert back to matrix
            c2w_new_odom[:3, :3] = new_rot_odom.as_matrix()

            c2w_odom = c2w_new_odom

            # print("c2w odom shape: ", c2w_odom.shape)

            odoms.append(c2w_odom.astype(np.float32))

            # print("c2w odom as type float32: ", c2w_odom.astype(np.float32))

        # print("first pose added: ", poses[0])
        # print("last pose added: ", poses[-1])
        # print("250th pose added: ", poses[249])

        print("number of poses added: ", len(poses))

        # poses_cf = transform_coordinate_system(poses)

        # print(odoms[10])
            
        return images, depths, poses, sync_timestamps
    

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, tstamp_odom, max_dt=10000000):
        # previous max_dt: 20000000
        """Pair images, depths, and poses based on timestamps
        
        Args:
            tstamp_image: timestamps from color images (nanoseconds)
            tstamp_depth: timestamps from depth images (nanoseconds)
            tstamp_pose: timestamps from pose data (nanoseconds)
            max_dt: maximum allowed time difference in nanoseconds
        
        Returns:
            List of tuples (color_idx, depth_idx, pose_idx) for synchronized frames
        """
        associations = []
        used_depth_indices = set()
        used_pose_indices = set()
        used_odom_indices = set()
        
        # Sort timestamps and keep track of original indices
        color_with_idx = [(t, i) for i, t in enumerate(tstamp_image)]
        depth_with_idx = [(t, i) for i, t in enumerate(tstamp_depth)]
        pose_with_idx = [(t, i) for i, t in enumerate(tstamp_pose)]
        odom_with_idx = [(t, i) for i, t in enumerate(tstamp_odom)]
        
        # Sort by timestamp
        color_with_idx.sort(key=lambda x: x[0])
        depth_with_idx.sort(key=lambda x: x[0])
        pose_with_idx.sort(key=lambda x: x[0])
        odom_with_idx.sort(key=lambda x: x[0])
        
        # Initialize pointers
        d_idx = 0
        p_idx = 0
        o_idx = 0

        d_diffs = []
        p_diffs = []
        o_diffs = []

        # print("number of odoms_with_idx: ", len(odom_with_idx))
        
        # For each color frame, find closest depth and pose and odom frames
        for color_time, color_idx in color_with_idx:
            # Move depth pointer until we're at or past the color timestamp
            d_idx_start = d_idx
            p_idx_start = p_idx
            o_idx_start = o_idx

            while (d_idx < len(depth_with_idx) - 1 and 
                depth_with_idx[d_idx][0] < color_time - max_dt):
                d_idx += 1
                
            # Move pose pointer until we're at or past the color timestamp
            while (p_idx < len(pose_with_idx) - 1 and 
                pose_with_idx[p_idx][0] < color_time - max_dt):
                p_idx += 1

            # Move pose pointer until we're at or past the color timestamp
            while (o_idx < len(odom_with_idx) - 1 and
                     odom_with_idx[o_idx][0] < color_time - max_dt):
                 o_idx += 1
            
            d_idx_end = d_idx
            p_idx_end = p_idx
            o_idx_end = o_idx

            d_diff = d_idx_end - d_idx_start
            p_diff = p_idx_end - p_idx_start
            o_diff = o_idx_end - o_idx_start

            d_diffs.append(d_diff)
            p_diffs.append(p_diff)
            o_diffs.append(o_diff)

            # Find best depth frame
            best_depth_idx = None
            min_depth_dt = float('inf')
            for i in range(max(0, d_idx - 10), min(len(depth_with_idx), d_idx + 20)):
                if i in used_depth_indices:
                    continue
                dt = abs(depth_with_idx[i][0] - color_time)
                if dt < min_depth_dt and dt < max_dt:
                    min_depth_dt = dt
                    best_depth_idx = i
                    
            # Find best pose frame
            best_pose_idx = None
            min_pose_dt = float('inf')
            for i in range(max(0, p_idx - 10), min(len(pose_with_idx), p_idx + 20)):
                if i in used_pose_indices:
                    continue
                dt = abs(pose_with_idx[i][0] - color_time)
                if dt < min_pose_dt and dt < max_dt:
                    min_pose_dt = dt
                    best_pose_idx = i

            # find best odom frame
            best_odom_idx = None
            min_odom_dt = float('inf')
            for i in range(max(0, o_idx - 10), min(len(odom_with_idx), o_idx + 20)):
                if i in used_odom_indices:
                    continue
                dt = abs(odom_with_idx[i][0] - color_time)
                if dt < min_odom_dt and dt < max_dt:
                    min_odom_dt = dt
                    best_odom_idx = i
                    
            # If we found matching frames within max_dt, add to associations
            if best_depth_idx is not None and best_pose_idx is not None and best_odom_idx is not None:
                associations.append((
                    color_idx,
                    depth_with_idx[best_depth_idx][1],
                    pose_with_idx[best_pose_idx][1],
                    # print("pose with idx: ", pose_with_idx[best_pose_idx][1]),
                    # print("odom with best idx: ", odom_with_idx[best_odom_idx][1]),
                    odom_with_idx[best_odom_idx][1]
                ))
                used_depth_indices.add(best_depth_idx)
                used_pose_indices.add(best_pose_idx)
                used_odom_indices.add(best_odom_idx)
            
        # TODO: Check time difference at best indices
        # print("average d_diff: ", sum(d_diffs) / len(d_diffs))
        # print("average p_diff: ", sum(p_diffs) / len(p_diffs))
        # print("average o_diff: ", sum(o_diffs) / len(o_diffs))

        print(f"Successfully associated {len(associations)} frames")

        return associations

    def pose_matrix_from_position_quaternion(self, position, quaternion):
        """Convert position and quaternion to 4x4 matrix
        
        Args:
            position: numpy array [x, y, z]
            quaternion: numpy array [qw, qx, qy, qz]
        """
        from scipy.spatial.transform import Rotation
        
        pose = np.eye(4)
        # Set translation directly
        pose[:3, 3] = position
        
        # Convert quaternion to rotation matrix
        # Note: Rotation.from_quat expects [qx, qy, qz, qw] order
        quat_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        pose[:3, :3] = Rotation.from_quat(quat_scipy).as_matrix()
        
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))

        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            
        return index, color_data, depth_data, self.poses[index]
        # return index, color_data, depth_data, self.poses[index], None


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "color").glob("*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(self.dataset_path / "pose")

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(path.glob('*.txt'),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                ls.append(list(map(float, line.split(' '))))
            c2w = np.array(ls).reshape(4, 4).astype(np.float32)
            self.poses.append(c2w)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, None, self.poses[index]


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, None, self.poses[index]


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == "kimera":
        return Kimera
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")

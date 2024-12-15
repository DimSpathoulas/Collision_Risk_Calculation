import argparse
import json
from dataclasses import dataclass
from typing import List
from torch import nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import norm
from tqdm import tqdm

from tools.TrackingBox import TrackingBox
from cov_cent import Covariance
from tools.data_classes import EvalBoxes
from tools.minkowskisum import minkowskisum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### VIZUALIZER DOES NOT CORRENTLY WORK
def vizualizer(pt, sample_csp, relative_pos, relative_rot, obstacles, curr_sample, distance_t):
    # Clear the current axis
    ax.clear()

    ego_rect_patch = patches.Polygon(pt, edgecolor='red', facecolor='none')

    if sample_csp < 0.2:
        ego_rect_patch.set_edgecolor("green")
    if 0.2 <= sample_csp < 0.34:
        ego_rect_patch.set_edgecolor("blue")
    if 0.34 <= sample_csp <= 0.50:
        ego_rect_patch.set_edgecolor("yellow")

    ax.add_patch(ego_rect_patch)
    ax.text(0, 0, f'{sample_csp:.2f}', color='black', ha='center', va='center')

    # Plot track rectangles with labels

    if obstacles is not None:
        for obs in obstacles:
            track_corners = obstacles[obs]['corners']
            track_corners = track_corners - relative_pos
            track_corners = (relative_rot @ track_corners.T).T
            track_rect_patch = patches.Polygon(track_corners, edgecolor='red', facecolor='none')

            csp = obstacles[obs]['csp']
            if csp < 0.2:
                track_rect_patch.set_edgecolor('green')
            if 0.2 <= csp < 0.34:
                track_rect_patch.set_edgecolor('blue')
            elif 0.34 <= csp <= 0.50:
                track_rect_patch.set_edgecolor('yellow')

            ax.add_patch(track_rect_patch)

            # clatter
            # label_x, label_y = np.mean(track_corners[:, 0]), np.mean(track_corners[:, 1])
            # ax.text(label_x, label_y, obs, color='black', ha='center', va='center')

    d = distance_t + distance_t * 0.5
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)

    ax.set_aspect('equal', adjustable='box')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Display the plot
    plt.title(f'Token {curr_sample}')
    plt.pause(0.5)


def rotz(t):
    ''' Rotation about the z-axis. '''

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s, c]])


def state_vector(translation, rotation, size, velocity):
    # [x, y, z, rot_z, w, l, h, vel_x, vel_y ]
    # [0, 1, 2,   3  , 4, 5, 6,   7  ,   8 ]
    q = Quaternion(rotation)
    box_rotation = q.angle if q.axis[2] > 0 else -q.angle  # clip angle

    temp_box = np.array([
        translation[0], translation[1], translation[2], box_rotation,
        size[0], size[1], size[2], velocity[0], velocity[1]])

    return temp_box


def get_2d_corners(bbox3d):
    """
    [x,  y,  z,  rot_z, l, w , h, vel_x, vel_y]
    x -> l , y -> w
    1----------0
    -          -
    -          -
    -          -
    -          -
    -          -
    -          -
    2----------3
    """
    p0 = np.array([bbox3d[4] / 2, bbox3d[5] / 2])
    p1 = np.array([-p0[0], p0[1]])
    p2 = np.array([p1[0], -p1[1]])
    p3 = np.array([p0[0], p2[1]])

    # Apply rotation
    R = rotz(bbox3d[3])
    corners = np.array([p0, p1, p2, p3])
    corners = (R @ corners.T).T
    # Translate back to original position
    corners[:, 0] += bbox3d[0]
    corners[:, 1] += bbox3d[1]
    return corners


def create_octagon(ego_rect, obstacle_rect):
    octagon = minkowskisum(ego_rect, obstacle_rect)

    octagon_centroid = np.mean(octagon, axis=0)
    ego_position = np.mean(ego_rect, axis=0)
    translation_vector = ego_position - octagon_centroid
    octagon = octagon + translation_vector  # center of octagon is center of ego
    return octagon


@dataclass
class OctagonEdge:
    xl: float
    xu: float
    m: float
    b: float
    contribution: int = 1  # 1 for positive, -1 for negative

def split_octagon_segments(octagon_vertices: np.ndarray) -> List[OctagonEdge]:
    # Get centroid
    centroid = np.mean(octagon_vertices, axis=0)

    # Sort vertices by angle around centroid
    angles = np.arctan2(octagon_vertices[:, 1] - centroid[1],
                        octagon_vertices[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_vertices = octagon_vertices[sorted_idx]

    edges = []
    n = len(sorted_vertices)

    # Create edges while preserving octagon shape
    for i in range(n):
        x1, y1 = sorted_vertices[i]
        x2, y2 = sorted_vertices[(i + 1) % n]

        if abs(x2 - x1) < 1e-10:  # Vertical edge
            continue
            # m = 1e6 if y2 > y1 else -1e6
            # b = y1 - m * x1
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

        # Determine contribution based on midpoint position
        mid_y = (y1 + y2) / 2
        contribution = 1 if mid_y > centroid[1] else -1

        edges.append(OctagonEdge(
            xl=min(x1, x2),
            xu=max(x1, x2),
            m=m,
            b=b,
            contribution=contribution
        ))

    return edges


#################################
# CSP CALCULATOR
#################################

def transform_to_obstacle_centered(points: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Transform points to obstacle-centered coordinates."""

    # relative position
    translated = points - pos

    # Inverse = Transposed rotation
    c, s = np.cos(-rot), np.sin(-rot)
    R = np.array([[c, -s], [s, c]])

    return translated @ R.T


def motion_prediction_ego(ego, sample, records, t, nusc, last_sample_token):
    """
    return: a dictionary of timestamps containing in each the position and velocity
    DISTRICT TIME IS 0.5 seconds -> 1 Timestamp
    """
    # [x, y, z, rot_z, w, l, h, vel_x, vel_y, x_cov, y_cov]
    # [0, 1, 2,   3  , 4, 5, 6,   7  ,   8  ,   9  ,   10 ]

    ego_predict = {}
    i = 0
    ego_predict[i] = ego
    size_eg = [ego[4], ego[5], ego[6]]

    while i < t and sample != last_sample_token:
        i = i + 1
        velocity_ego = [0, 0]
        sample = nusc.get('sample', sample)['next']
        sample_record = nusc.get('sample', sample)
        timestamp = sample_record['timestamp']

        ego_pose = next((pose for pose in records if pose['timestamp'] == timestamp), None)

        translation_ego = ego_pose['translation']  # [x, y, z]
        rotation_ego = ego_pose['rotation']  # [qw, qx, qy, qz]

        velocity_ego[0] = (translation_ego[0] - ego_predict[i - 1][0]) / 0.5
        velocity_ego[1] = (translation_ego[1] - ego_predict[i - 1][1]) / 0.5

        e = state_vector(translation=translation_ego, rotation=rotation_ego, size=size_eg,
                         velocity=velocity_ego)

        ego_predict[i] = e

    ego_projections = {
        i: get_2d_corners(ego_predict[i])
        for i in ego_predict
    }

    return ego_projections


def predict_state_and_covariance(statet, covt, Q, t=0.5, base_interval=0.5):
    """Predict state and covariance at time t"""
    F = np.array([[1, 0, t, 0],
                  [0, 1, 0, t],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    scaled_Q = Q * (t / base_interval)

    # Predict state and covariance
    state_t = F @ statet
    cov_t = F @ covt @ F.T + scaled_Q

    return state_t, cov_t


class CSPCalculator:
    def __init__(self, sigma_x: float, sigma_y: float):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sqrt2 = np.sqrt(2)

    def update_cov(self, sigma_x=None, sigma_y=None):
        """Update parameters."""
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def _integrand(self, x: float, m: float, b: float) -> float:
        """Compute integrand according to equation (5)."""
        erf_term = erf((m * x + b) / (self.sqrt2 * self.sigma_y))
        exp_term = np.exp(-0.5 * (x / self.sigma_x) ** 2)
        return erf_term * exp_term

    def calculate_segment_probability(self, edge: OctagonEdge) -> float:
        norm_factor = 1.0 / (np.sqrt(8 * np.pi) * self.sigma_x)

        contribution = float(edge.contribution)

        results = quad(
            lambda x: self._integrand(x, edge.m, edge.b),
            edge.xl,
            edge.xu
        )

        result = results[0] - results[1]

        return norm_factor * result * contribution

    def calculate_csp(self, octagon: np.ndarray, obstacle_pos: np.ndarray,
                      obstacle_rot: np.ndarray) -> float:
        transformed_octagon = transform_to_obstacle_centered(
            octagon, obstacle_pos, obstacle_rot)
        edges = split_octagon_segments(transformed_octagon)

        total_csp = 0.0
        for edge in edges:
            total_csp += self.calculate_segment_probability(edge)
        
        # if total_csp < -0.001 or total_csp > 1.001:
        #     print('hmm', total_csp)
            
        return max(0.0, min(1.0, total_csp))


def CSP(ego_rectangles, track, P, Q, spat, Net, crit, training):

    loss = None

    state_inds = [0, 1, 7, 8]
    tracks = {}
    obstacle_rects = {}
    Pts = {}
    state_t = track[state_inds]
    P_t = P

    tracks[0] = track
    obstacle_rects[0] = get_2d_corners(track)
    Pts[0] = P_t

    for i in range(1, len(ego_rectangles)):
        statet, Pt = predict_state_and_covariance(state_t, P_t, Q)
        state_t = statet
        P_t = Pt

        updated_track = track.copy()
        updated_track[state_inds] = state_t
        tracks[i] = updated_track
        obstacle_rects[i] = get_2d_corners(bbox3d=tracks[i])
        Pts[i] = P_t


    csp_results = []

    for i in range(len(ego_rectangles)):

        octagon = create_octagon(ego_rectangles[i], obstacle_rects[i])

        if i == 0:
            csp_calculator = CSPCalculator(sigma_x=float(Pts[i][0, 0]), sigma_y=float(Pts[i][1, 1]))
        else:
            csp_calculator.update_cov(sigma_x=float(Pts[i][0, 0]), sigma_y=float(Pts[i][1, 1]))

        csp = csp_calculator.calculate_csp(octagon, tracks[i][:2], tracks[i][3])
        csp_results.append(csp)

    csp_results = torch.tensor(csp_results).unsqueeze(1).to(device=device).float()

    if training:
        csp_estimations = Net(ego_rectangles, obstacle_rects, Pts)
    else:
        with torch.no_grad():
            csp_estimations = Net(ego_rectangles, obstacle_rects, Pts)
            if sum(csp_results) > 0.1:
                print(csp_estimations, csp_results)
                loss = crit(csp_estimations, csp_results)
                print(loss)

    if training:  # hard mining?
        loss = crit(csp_estimations, csp_results)

    return csp_results, csp_estimations, loss


class CSP_NET(nn.Module):
    def __init__(self):
        super(CSP_NET, self).__init__()

        # 4,2 4,2 rectangles and 2 from P
        self.net = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, in1, in2, in3):

        x1 = torch.tensor(np.array(list(in1.values()))).to(device=device)
        x2 = torch.tensor(np.array(list(in2.values()))).to(device=device)
        x3_inter = torch.tensor(np.array(list(in3.values()))).to(device=device)

        p00 = x3_inter[:, 0, 0]
        p11 = x3_inter[:, 1, 1]

        x3 = torch.stack((p00, p11), dim=1)

        # Flatten each input
        x1_flat = x1.view(x1.size(0), -1)  # (n, 8)
        x2_flat = x2.view(x2.size(0), -1)  # (n, 8)
        x3_flat = x3.view(x3.size(0), -1)  # (n, 2)

        # Concatenate along dimension 1 (n, 18)
        y = torch.cat((x1_flat, x2_flat, x3_flat), dim=1).float()

        # Pass the concatenated tensor through the network
        z = self.net(y)

        return z


def save_tracker_states(csp_net, save_path):
    state_dict = csp_net.state_dict()
    torch.save(state_dict, save_path)
    print(f"Saved csp net to {save_path}")


def load_tracker_states(csp_net, load_path):

    if os.path.exists(load_path):
        state_dict = torch.load(load_path)
        csp_net.load_state_dict(state_dict)
        print(f"Loaded CSP_NET state from {load_path}")

    else:
        print(f"No saved state found at {load_path}")

    return csp_net



def collision_risk_calculation():
    args = parse_arguments()

    CSP_Net = CSP_NET().to(device)

    training = args.training
    criterion = nn.BCELoss()

    if training:
        params_to_optimize = list(CSP_Net.net.parameters())
        optimizer = torch.optim.Adam(params_to_optimize, lr=0.001)
        EPOCHS = 5

    else:
        CSP_Net = load_tracker_states(CSP_Net, load_path=args.load_path)

        for param in CSP_Net.net.parameters():
            param.requires_grad = False
            
        CSP_Net.net.eval()

        EPOCHS = 1

    size_ego = [4.084 // 2 + 0.5  ,1.73 , 1.0]

    data_root = args.data_root
    version = args.version
    tracking_file = args.tracking_file
    distance_thresh = args.distance_thresh
    projection_window = args.projection_window * 2

    # Read data from the tracking file
    with open(tracking_file) as f:
        data = json.load(f)

    assert 'results' in data, 'Error: No field `results` in result file.'

    all_results = EvalBoxes.deserialize(data['results'], TrackingBox)

    # meta = data['meta']
    # print('meta: ', meta)
    # print("Loaded results from {}. Found trackers for {} samples."
    #       .format(tracking_file, len(all_results.sample_tokens)))

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    results = {}

    total_time = 0.0
    total_frames = 0

    time_frame = 0.5  # roughly 0.5 seconds per sample

    print(device)
    ego_pose_records = nusc.ego_pose
    tracks_list = {}
    tracks_info = {}
    covariance = Covariance()

    data_dict = {}
    fig, ax = plt.subplots()


    for epoch in range(EPOCHS):

        print('epoch', epoch + 1)

        processed_scene_tokens = set()
        for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
            sample_token = all_results.sample_tokens[sample_token_idx]
            scene_token = nusc.get('sample', sample_token)['scene_token']
            if scene_token in processed_scene_tokens:
                continue

            # print("\nScene", scene_token)

            first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
            last_sample_token = nusc.get('scene', scene_token)['last_sample_token']
            current_sample_token = first_sample_token

            processed_scene_tokens.add(scene_token)

            while current_sample_token != last_sample_token:

                sample_loss = None
    
                # print(current_sample_token)
                sample_record = nusc.get('sample', current_sample_token)
                sample_timestamp = sample_record['timestamp']

                # Filter ego_pose_records based on the sample timestamp
                ego_pose = next((pose for pose in ego_pose_records if pose['timestamp'] == sample_timestamp), None)
                # ego_pose = [pose for pose in ego_pose_records if pose['timestamp'] == sample_timestamp]
                translation_ego = ego_pose['translation']  # [x, y, z]
                rotation_ego = ego_pose['rotation']  # [qw, qx, qy, qz]

                if first_sample_token == current_sample_token:
                    velocity_ego = [0, 0]
                    ego_box = state_vector(translation=translation_ego, rotation=rotation_ego, size=size_ego,
                                        velocity=velocity_ego)

                # could be accel model for better results
                velocity_ego[0] = (translation_ego[0] - ego_box[0]) / time_frame
                velocity_ego[1] = (translation_ego[1] - ego_box[1]) / time_frame

                ego_box = state_vector(translation=translation_ego, rotation=rotation_ego, size=size_ego,
                                    velocity=velocity_ego)

                ego_rectangles = motion_prediction_ego(ego_box, current_sample_token, ego_pose_records, projection_window,
                                                    nusc, last_sample_token)
                obstacles = {}
                cs = 0.0

                # for every track
                for box in all_results.boxes[current_sample_token]:
                    # [x, y, z, rot_z, w, l, h, vel_x, vel_y]
                    # [0, 1, 2,   3  , 4, 5, 6,   7  ,   8 ]

                    track = state_vector(translation=box.translation, rotation=box.rotation, size=box.size,
                                        velocity=box.velocity)

                    # if in distance threshold
                    current_distance = np.sqrt(((ego_box[0] - track[0]) ** 2) + ((ego_box[1] - track[1]) ** 2))
                    if current_distance > distance_thresh:
                        continue

                    # GET VARIANCES
                    # we need x, y, l, w, vel_x, vel_y, (rot_z) Ps and Qs
                    P, Q, spat = covariance.get_S(box.tracking_name)
    
                    csp_analytical, csp_network, loss = CSP(ego_rectangles, track, P, Q, spat, 
                                                            CSP_Net, criterion, training)
                    
                    if loss is not None:
                        if sample_loss is None:
                            sample_loss = loss
                        else:
                            sample_loss = loss + sample_loss
                            
                        cs = cs + 1.0

                    # entry = {
                    #         'csp': csp,
                    #         'name': box.tracking_name,
                    # }
                    # obstacles[box.tracking_id] = entry

                ego_csp = max((entry['csp'] for entry in obstacles.values()), default=0)
                
                current_sample_token = nusc.get('sample', current_sample_token)['next']

                if training and sample_loss is not None:
                    optimizer.zero_grad()
                    sample_loss.div(cs)
                    sample_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                    optimizer.step()

                else:
                    pass

    save_tracker_states(CSP_Net, args.save_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collision Risk Calculator")
    parser.add_argument("--data_root", type=str, default='/second_ext4/ktsiakas/kosmas/nuscenes/v1.0-trainval',
                        help="Path to the data root directory")
    parser.add_argument("--version", type=str, default='v1.0-trainval',
                        help="Version (e.g., v1.0-mini)")
    parser.add_argument("--tracking_file", type=str, default='/home/ktsiakas/thesis_new/PROB_3D_MULMOD_MOT/results_val_probabilistic_tracking.json',
                        help="Path to the tracking file")
    
    parser.add_argument("--distance_thresh", type=float, default= 10,
                        help="Distance threshold (in meters)")
    parser.add_argument("--projection_window", type=float, default= 2,
                help="Model prediction (in seconds)")
    
    parser.add_argument("--training", type=str, default= False,
                help="Model train = True or val")
    parser.add_argument("--save_path", type=str, default= 'model_10m_2s.pth',
                help="save the model if in train")
    parser.add_argument("--load_path", type=str, default= 'model_10m_2s.pth',
                help="load the model if in val")
    
    return parser.parse_args()


if __name__ == '__main__':

    collision_risk_calculation()
 

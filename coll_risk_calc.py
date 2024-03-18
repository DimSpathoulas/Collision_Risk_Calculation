# SPATHOULAS DIMITRIS FALL 2023
# Module heavily based on: https://ieeexplore.ieee.org/document/8793264


'''
Run:
python coll_risk_calc.py
 --data_root Path to nuscenes dataset version
 --version The version
 --tracking_file Path to 3d detectors outputs
 --distance_thresh Absolute distance between ego and track centers - Compute CSPs
 --seconds_to_prediction Seconds of prediction


 Example:
 python coll_risk_calc.py
 --data_root data\nusc_mini
 --version v1.0-mini
 --tracking_file data\tracking\results_minitrain_probabilistic_tracking.json
 --distance_thresh 15
 --seconds_to_prediction 4
 '''


import json
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
import argparse
from tools.TrackingBox import TrackingBox
from tools.covariance import Covariance
from tools.data_classes import EvalBoxes
from tqdm import tqdm
from tools.minkowskisum import minkowskisum
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def vizualizer(pt, sample_csp, relative_pos, relative_rot, obstacles, curr_sample, distance_t):
    ''' Vizualizes results. '''

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

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.title(f'Token {curr_sample}')
    plt.pause(0.3)


def rotz(t):
    ''' Rotation about the z-axis. '''

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s, c]])


def create_box(translation, rotation, size, velocity, covar=None):
    '''
    Create boxes
     [x, y, z,  rot_z, l, w, h, vel_x, vel_y, cov_x, cov_y]
     [0, 1, 2,    3,   4, 5, 6,   7,     8,     9,    10  ]
     '''

    q = Quaternion(rotation)
    box_rotation = q.angle if q.axis[2] > 0 else -q.angle

    if covar is not None:
        s_1 = S[0, 0]
        s_2 = S[1, 1]
        temp_box = np.array([
            translation[0], translation[1], translation[2], box_rotation,
            size[1], size[0], size[2], velocity[0], velocity[1], s_1, s_2])
    else:
        temp_box = np.array([
            translation[0], translation[1], translation[2], box_rotation,
            size[1], size[0], size[2], velocity[0], velocity[1], 0, 0])

    return temp_box


def get_2d_corners(bbox3d):
    """
    Create a BEV rectangle
    [x,  y,  z,  rot_z, w, l , h, vel_x, vel_y]
    x -> l , y -> w
    1----------0
    -          -
    -          -
    -          -
    2----------3
    """

    p0 = np.array([bbox3d[4] / 2, bbox3d[5] / 2])
    p1 = np.array([-p0[0], p0[1]])
    p2 = np.array([p1[0], -p1[1]])
    p3 = np.array([p0[0], p2[1]])

    R = rotz(bbox3d[3])
    corners = np.array([p0, p1, p2, p3])
    corners = (R @ corners.T).T

    corners[:, 0] += bbox3d[0]
    corners[:, 1] += bbox3d[1]
    return corners


def motion_prediction_ego(ego, sample, records, t, nusc):
    """
    Return: a dictionary of timestamps containing in each the position and velocity
    THIS IS A VERY SIMPLE MOTION MODEL
    DISTRICT TIME IS 0.5 seconds -> 1 Timestamp
    """

    ego_predict = {}
    steps = 0
    time_frame = 0.5
    sample_record = nusc.get('sample', sample)
    timer = sample_record['timestamp']
    ego_predict[timer] = ego
    sample = nusc.get('sample', sample)['next']

    while steps < t:

        try:
            sample_record = nusc.get('sample', sample)
            timestamp = sample_record['timestamp']
            ego_pose = [pose for pose in records if pose['timestamp'] == timestamp]
            ego_pose = ego_pose[0]
            translation_ego = ego_pose['translation']  # [x, y, z]
            rotation_ego = ego_pose['rotation']  # [qw, qx, qy, qz]

            velocity_ego[0] = (translation_ego[0] - ego_predict[timer][0]) / time_frame  # no need
            velocity_ego[1] = (translation_ego[1] - ego_predict[timer][1]) / time_frame  # no need
            size_eg = [ego[4], ego[5], ego[6]]
            e = create_box(translation=translation_ego, rotation=rotation_ego, size=size_eg,
                           velocity=velocity_ego)
            steps = steps + 1
            timer = timestamp
            ego_predict[timer] = e
            sample = nusc.get('sample', sample)['next']

        except KeyError:
            break

    return ego_predict, steps


def motion_prediction_track(track, timesamples):
    '''
    Motion ego prediction
    Deterministic
    '''

    iterations = len(timesamples)
    track_predict = {}
    track_predict[timesamples[0]] = track
    time_step = 0.5
    constant_velocity = np.array([track[7], track[8]])
    new_track = track.copy()
    prior_state = np.array([new_track[0], new_track[1]])

    for i in range(1, iterations):
        adding_vector = constant_velocity * time_step
        posterior_state = prior_state + adding_vector
        new_track[0:2] = posterior_state
        track_predict[timesamples[i]] = new_track

        time_step += 0.5

    return track_predict


def octagon_creation(ego_rect, track_poses):
    '''Octagon creaction
    For each track - ego pair of n poses ( predictions )
    compute the minkowski sum
    The convolution between two rectangles will be an octagon
    With center, the center of ego
    '''

    octagons = {}

    for i, timesample in enumerate(ego_rect):
        curr_ego_rect = ego_rect[timesample]
        track_rect = get_2d_corners(track_poses[timesample])

        octagon = minkowskisum(curr_ego_rect, track_rect)
        octagon_centroid = np.mean(octagon, axis=0)
        ego_position = np.mean(curr_ego_rect, axis=0)
        translation_vector = ego_position - octagon_centroid
        octagon = octagon + translation_vector
        octagons[timesample] = octagon

    return octagons


def csp_calculation(octs, tracks):
    '''Compute the CSP of n predictions
    Keep the biggest
    The CSP is computed as the volume under the area of intersection between the octagon ( ego )
    and a gaussian of center (0,0) (track) with covariance (size + cov).
    '''

    csp = -1.0

    def gaussian_pdf(x, y):
        return gaussian.pdf([x, y])

    for i, timesample in enumerate(tracks):

        h = (tracks[timesample][9] + tracks[timesample][4])
        k = (tracks[timesample][10] + tracks[timesample][5])
        covariance = np.array([[abs(h), 0], [0, abs(k)]])

        gaussian = multivariate_normal(mean=(0, 0), cov=covariance)
        octagon = octs[timesample]
        relative_pos = [tracks[timesample][0], tracks[timesample][1]]

        R = rotz(tracks[timesample][3]).T
        octagon = octagon - relative_pos
        octagon = (R @ octagon.T).T

        # The dblquad creates a rectangular box from the octagon
        # A more accurate method for computing the intersection between
        # The octagon and gaussian will be integrated in the future

        probability, error = dblquad(gaussian_pdf, octagon[:, 0].min(), octagon[:, 0].max(),
                                     lambda x: octagon[:, 1].min(), lambda y: octagon[:, 1].max())

        cur_csp = probability - error
        if cur_csp > csp:
            csp = cur_csp

    return csp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Collision Risk Calculator")
    parser.add_argument("--data_root", type=str, help="Path to the data root directory")
    parser.add_argument("--version", type=str, help="Version (e.g., v1.0-mini)")
    parser.add_argument("--tracking_file", type=str, help="Path to the tracking file")
    parser.add_argument("--distance_thresh", type=float, help="Distance threshold (in meters)")
    parser.add_argument("--seconds_to_prediction", type=float, help="Seconds of prediction")
    return parser.parse_args()



if __name__ == '__main__':

    args = parse_arguments()

    data_root = args.data_root
    version = args.version
    tracking_file = args.tracking_file
    distance_thresh = args.distance_thresh
    seconds_to_prediction = args.seconds_to_prediction

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    with open(tracking_file) as f:
        data = json.load(f)

    assert 'results' in data, 'Error: No field `results` in result file.'

    all_results = EvalBoxes.deserialize(data['results'], TrackingBox)

    meta = data['meta']
    # print('meta: ', meta)
    # print("Loaded results from {}. Found trackers for {} samples."
    #       .format(tracking_file, len(all_results.sample_tokens)))

    results = {}

    total_time = 0.0
    total_frames = 0

    t = seconds_to_prediction * 2
    w_ego = 1.73
    l_ego = 4.084 // 2 + 0.5
    h_ego = 1.0
    time_frame = 0.5
    size_ego = [w_ego, l_ego, h_ego]
    ego_pose_records = nusc.ego_pose
    tracks_list = {}
    tracks_info = {}
    covariance = Covariance()

    data_dict = {}
    fig, ax = plt.subplots()
    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue

        print("\nScene", scene_token)

        first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        last_sample_token = nusc.get('scene', scene_token)['last_sample_token']
        current_sample_token = first_sample_token

        while current_sample_token != last_sample_token:

            sample_record = nusc.get('sample', current_sample_token)
            sample_timestamp = sample_record['timestamp']

            ego_pose = [pose for pose in ego_pose_records if pose['timestamp'] == sample_timestamp]
            ego_pose = ego_pose[0]
            translation_ego = ego_pose['translation']
            rotation_ego = ego_pose['rotation']

            if first_sample_token == current_sample_token:
                velocity_ego = [0, 0]
                ego_box = create_box(translation=translation_ego, rotation=rotation_ego, size=size_ego,
                                     velocity=velocity_ego)

            velocity_ego[0] = (translation_ego[0] - ego_box[0]) / time_frame
            velocity_ego[1] = (translation_ego[1] - ego_box[1]) / time_frame

            ego_box = create_box(translation=translation_ego, rotation=rotation_ego, size=size_ego,
                                 velocity=velocity_ego)

            predicted_ego_motion, seconds = motion_prediction_ego(ego_box, current_sample_token, ego_pose_records, t, nusc)

            key_timesamples = list(predicted_ego_motion.keys())

            ego_rect_dc = {}
            for i, timesample in enumerate(predicted_ego_motion):
                ego_rect_dc[timesample] = get_2d_corners(predicted_ego_motion[timesample])

            obstacles = {}

            for box in all_results.boxes[current_sample_token]:

                S = covariance.get_S(box.tracking_name)
                track = create_box(translation=box.translation, rotation=box.rotation, size=box.size,
                                   velocity=box.velocity, covar=S)

                current_distance = np.sqrt(((ego_box[0] - track[0]) ** 2) + ((ego_box[1] - track[1]) ** 2))

                if current_distance < distance_thresh:

                    predicted_track_motion = motion_prediction_track(track, key_timesamples)
                    octagons = octagon_creation(ego_rect_dc, predicted_track_motion)

                    csp = csp_calculation(octagons, predicted_track_motion)

                    entry = {
                        'csp': csp,
                        'name': box.tracking_name,
                        'corners': get_2d_corners(predicted_track_motion[sample_timestamp])
                    }
                    obstacles[box.tracking_id] = entry

                else:
                    continue

            ego_csp = max((entry['csp'] for entry in obstacles.values()), default=0)

            print("collision_prob {:.4f} of sample {} with {} tracks"
                  .format(ego_csp, current_sample_token, len(obstacles)))

            # Can be saved for further calculations and modules
            # data_dict[current_sample_token] = {
            #     'ego_box': ego_box,
            #     'ego_csp': ego_csp,
            #     'obstacles': obstacles
            # }

            ego_rect = np.concatenate(ego_rect_dc[sample_timestamp], axis=0).reshape(4, 2)
            ego_position = np.mean(ego_rect, axis=0)
            p = ego_rect - ego_position
            r = rotz(ego_box[3]).T
            p = (r @ p.T).T

            vizualizer(p, ego_csp, ego_position, r, obstacles, current_sample_token, distance_thresh)

            current_sample_token = nusc.get('sample', current_sample_token)['next']
            processed_scene_tokens.add(scene_token)

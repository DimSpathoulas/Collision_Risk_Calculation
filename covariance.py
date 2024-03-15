import numpy as np


class Covariance(object):
    def __init__(self):

        NUSCENES_TRACKING_NAMES = [
            'bicycle',
            'bus',
            'car',
            'motorcycle',
            'pedestrian',
            'trailer',
            'truck'
        ]
        # Kalman Filter state: [x, y, z, rot_z, l, w, h, x_dot, y_dot, z_dot, rot_z_dot]
        #
        P = {  # x + w, y + l - the center variance + the size variance in that dimension
            'bicycle':    [0.05390982 + 0.02713823, 0.05039431 + 0.01169572],  # 1.29464435, 0.02713823
            'bus':        [0.17546469 + 0.78867322, 0.13818929 + 0.05507407],  # 0.1979503 , 0.78867322,
            'car':        [0.08900372 + 0.10912802, 0.09412005 + 0.02359175],  # 1.00535696, 0.10912802
            'motorcycle': [0.04052819 + 0.03291016, 0.03989040 + 0.00957574],  # 1.06442726, 0.03291016
            'pedestrian': [0.03855275 + 0.02286483, 0.03771110 + 0.0136347],  # 2.0751833 , 0.02286483
            'trailer':    [0.23228021 + 1.37451601, 0.22229261 + 0.06354783],  # 1.05163481, 1.37451601,
            'truck':      [0.14862173 + 0.69387238, 0.14445960 + 0.05484365]  # 0.73122169, 0.69387238,
        }

        self.P = {tracking_name: np.diag(P[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}

    def get_S(self, tracking_name):
        if tracking_name in self.P:
            return self.P[tracking_name]
        else:
            print(f"Invalid tracking name: {tracking_name}")
            return None

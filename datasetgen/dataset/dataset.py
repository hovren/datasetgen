from __future__ import print_function, division

import os
from collections import namedtuple
import bisect

import h5py
import numpy as np
import crisp.rotations
import crisp.fastintegrate
from ..maths.quaternions import Quaternion, QuaternionArray
from datasetgen.utilities.time_series import TimeSeries
from datasetgen.trajectories.splined import \
    SplinedPositionTrajectory, SampledPositionTrajectory, \
    SampledRotationTrajectory, SplinedRotationTrajectory, \
    SampledTrajectory, SplinedTrajectory

from .utils import resample_quaternion_array, create_bounds


class DatasetError(Exception):
    pass

class Landmark(object):
    __slots__ = ('id', '_color', 'position', 'visibility', '_observations')

    def __init__(self, _id, position, observations, color=None):
        self.id = _id
        self.position = position
        self._color = color
        self.visibility = None # Set by observation setter
        self.observations = observations

    def __repr__(self):
        return '<Landmark #{id:d} ({X[0]:.2f}, {X[1]:.2f}, {X[2]:.2f})>'.format(id=self.id, X=self.position)

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, obs):
        if isinstance(obs, dict):
            self._observations = obs
            self.visibility = set(obs.keys())
        else:
            self._observations = {view_id : None for view_id in obs}
            self.visibility = set(obs)

    @property
    def color(self):
        if self._color is None:
            return 255 * np.ones((4,), dtype='uint8')
        else:
            r, g, b = self._color[:3]
            a = 255 if self._color.size == 3 else self._color[-1]
            return np.array([r, g, b, a], dtype='uint8')


class Dataset(object):
    def __init__(self):
        self._position_data = None
        self._orientation_data = None
        self.trajectory = None
        self.landmarks = []
        self._landmark_bounds = None
        self.name = None

    def set_position_data(self, times, data):
        self._position_data = TimeSeries(times, data)
        self._update_trajectory()

    def set_orientation_data(self, times, data):
        self._orientation_data = TimeSeries(times, data)
        self._update_trajectory()

    def set_landmarks(self, view_times, landmarks):
        if not np.all(np.diff(view_times) > 0):
            raise DatasetError("View times are not increasing monotonically with id")
        self._landmark_bounds = create_bounds(view_times)
        self.landmarks.clear()
        self.landmarks.extend(landmarks)

    def visible_landmarks(self, t):
        i = bisect.bisect_left(self._landmark_bounds, t)
        interval_id = i - 1
        return [lm for lm in self.landmarks if interval_id in lm.visibility]

    def save(self, filepath, name):
        if os.path.exists(filepath):
            raise DatasetError('File {} already exists'.format(filepath))
        with h5py.File(filepath, 'w') as h5f:
            def save_keyframes(ts, groupkey):
                group = h5f.create_group(groupkey)
                if isinstance(ts.values, QuaternionArray):
                    values = ts.values.array
                else:
                    values = ts.values
                group['data'] = values
                group['timestamps'] = ts.timestamps

            h5f.attrs['name'] = name

            save_keyframes(self._position_data, 'position')
            save_keyframes(self._orientation_data, 'orientation')

            landmarks_group = h5f.create_group('landmarks')
            landmarks_group['visibility_bounds'] = self._landmark_bounds
            num_digits = int(np.ceil(np.log10(len(self.landmarks) + 0.5))) # 0.5 to avoid boundary conditions
            positions = np.empty((len(self.landmarks), 3))
            colors = np.empty_like(positions, dtype='uint8')
            visibility_group = landmarks_group.create_group('visibility')
            for i, landmark in enumerate(self.landmarks):
                vgroup_key = '{:0{pad}d}'.format(i, pad=num_digits)
                visibility_group[vgroup_key] = np.array(list(landmark.visibility)).astype('uint64')
                positions[i] = landmark.position
                colors[i] = landmark.color[:3] # Skip alpha
            landmarks_group['positions'] = positions
            landmarks_group['colors'] = colors

    def rescaled(self, scale_factor):
        ds_r = Dataset()
        ds_r._landmark_bounds = self._landmark_bounds
        ds_r._position_data = TimeSeries(self._position_data.timestamps,
                                         scale_factor * self._position_data.values)
        ds_r._orientation_data = TimeSeries(self._orientation_data.timestamps,
                                            self._orientation_data.values)
        ds_r._update_trajectory()

        for lm in self.landmarks:
            new_pos = scale_factor * lm.position
            lm_r = Landmark(lm.id, new_pos, lm.observations, color=lm.color)
            ds_r.landmarks.append(lm_r)

        return ds_r

    def rescaled_avg_speed(self, avg_speed):
        travel_time = self.trajectory.endTime - self.trajectory.startTime
        dt = 0.01 # seconds
        num_samples = travel_time / dt
        t = np.linspace(self.trajectory.startTime, self.trajectory.endTime, num=num_samples)
        velocity = self.trajectory.velocity(t) # Global frame, but OK since we want length only
        speed = np.linalg.norm(velocity, axis=0)
        distance = np.trapz(speed, dx=dt)
        scale = avg_speed * travel_time / distance
        return self.rescaled(scale)


    @classmethod
    def from_file(cls, filepath):
        instance = cls()

        def load_timeseries(group):
            timestamps = group['timestamps'].value
            data = group['data'].value
            if data.shape[1] == 4:
                data = QuaternionArray(data)
            return TimeSeries(timestamps, data)

        with h5py.File(filepath, 'r') as h5f:
            instance.name = h5f.attrs['name']
            instance._position_data = load_timeseries(h5f['position'])
            instance._orientation_data = load_timeseries(h5f['orientation'])
            instance._update_trajectory()

            landmarks_group = h5f['landmarks']
            instance._landmark_bounds = landmarks_group['visibility_bounds'].value
            positions = landmarks_group['positions'].value
            colors = landmarks_group['colors'].value
            landmark_keys = list(landmarks_group['visibility'].keys())
            landmark_keys.sort(key=lambda key: int(key))
            for lm_key in landmark_keys:
                lm_id = int(lm_key)
                p = positions[lm_id]
                color = colors[lm_id]
                visibility = set(list(landmarks_group['visibility'][lm_key].value))
                lm = Landmark(lm_id, p, visibility, color=color)
                instance.landmarks.append(lm)
        instance.landmarks = sorted(instance.landmarks, key=lambda lm: lm.id)

        return instance

    def _update_trajectory(self):
        smooth_rotations = False
        if self._position_data and not self._orientation_data:
            samp = SampledPositionTrajectory(self._position_data)
            self.trajectory = SplinedPositionTrajectory(samp)
        elif self._orientation_data and not self._position_data:
            samp = SampledRotationTrajectory(self._orientation_data)
            self.trajectory = SplinedRotationTrajectory(samp, smoothRotations=smooth_rotations)
        elif self._position_data and self._orientation_data:
            samp = SampledTrajectory(self._position_data, self._orientation_data)
            self.trajectory = SplinedTrajectory(samp, smoothRotations=smooth_rotations)

from __future__ import print_function, division

import os
from collections import namedtuple
import bisect

import h5py
import numpy as np
import crisp.rotations
import crisp.fastintegrate
from .maths.quaternions import Quaternion, QuaternionArray
from .utilities.time_series import TimeSeries
from .trajectories.splined import \
    SplinedPositionTrajectory, SampledPositionTrajectory, \
    SampledRotationTrajectory, SplinedRotationTrajectory, \
    SampledTrajectory, SplinedTrajectory


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

    def position_from_sfm(self, sfm):
        view_times = np.array([v.time for v in sfm.views])
        view_positions = np.vstack([v.position for v in sfm.views]).T
        ts = TimeSeries(view_times, view_positions)
        self._position_data = ts
        self._update_trajectory()

    def orientation_from_sfm(self, sfm):
        view_times = np.array([v.time for v in sfm.views])
        view_orientations = QuaternionArray([v.orientation for v in sfm.views])
        view_orientations = view_orientations.unflipped()
        # Resampling is important to get good splines
        view_orientations, view_times = resample_quaternion_array(view_orientations, view_times)
        ts = TimeSeries(view_times, view_orientations)
        self._orientation_data = ts
        self._update_trajectory()

    def orientation_from_gyro(self, gyro_data, gyro_times):
        n, d = gyro_data.shape

        if d == 3:
            dt = float(gyro_times[1] - gyro_times[0])
            if not np.allclose(np.diff(gyro_times), dt):
                raise DatasetError("gyro timestamps must be uniformly sampled")

            qdata = crisp.fastintegrate.integrate_gyro_quaternion_uniform(gyro_data, dt)
            Q = QuaternionArray(qdata)
        elif d == 4:
            Q = QuaternionArray(gyro_data)
        else:
            raise DatasetError("Gyro data must have shape (N,3) or (N, 4), was {}".format(gyro_data.shape))

        ts = TimeSeries(gyro_times, Q.unflipped())
        self._orientation_data = ts
        self._update_trajectory()

    def landmarks_from_sfm(self, sfm):
        view_times = [view.time for view in sfm.views]
        if not np.all(np.diff(view_times) > 0):
            raise DatasetError("View times are not increasing monotonically with id")

        self._landmark_bounds = create_bounds(view_times)
        for lm in sfm.landmarks:
            self.landmarks.append(lm)

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

    def visualize(self, draw_orientations=False, draw_axes=False):
        from mayavi import mlab
        t_min = self.trajectory.startTime
        t_max = self.trajectory.endTime
        t_samples = (t_max - t_min) * 50
        t = np.linspace(t_min, t_max, t_samples)
        positions = self.trajectory.position(t)
        landmark_data = np.vstack([lm.position for lm in self.landmarks]).T
        landmark_scalars = np.arange(len(self.landmarks))

        landmark_colors = np.vstack([lm.color for lm in self.landmarks])
        orientation_times = np.linspace(t_min, t_max, num=50)
        orientations = self.trajectory.rotation(orientation_times)

        # World to camera transform is
        # Xc = RXw - Rt where R is the camera orientation and position respectively
        # Camera to world is thus
        # Xw = RtXc + t
        zc = np.array([0, 0, 1.]).reshape(3,1)
        zw = [np.dot(np.array(q.toMatrix()).T, zc).reshape(3,1) for q in orientations]
        quiver_pos = self.trajectory.position(orientation_times)
        quiver_data = 0.5 * np.hstack(zw)

        pts = mlab.points3d(landmark_data[0], landmark_data[1], landmark_data[2],
                            landmark_scalars, scale_factor=0.2, scale_mode='none')
        pts.glyph.color_mode = 'color_by_scalar'
        pts.module_manager.scalar_lut_manager.lut.table = landmark_colors

        plot_obj = mlab.plot3d(positions[0], positions[1], positions[2], color=(1, 0, 0), line_width=5.0, tube_radius=None)
        if draw_axes:
            mlab.axes(plot_obj)
        if draw_orientations:
            mlab.quiver3d(quiver_pos[0], quiver_pos[1], quiver_pos[2],
                          quiver_data[0], quiver_data[1], quiver_data[2], color=(1, 1, 0))
        mlab.show()

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

from __future__ import print_function, division

import crisp.fastintegrate
import numpy as np
from ..maths.quaternions import QuaternionArray

from .dataset import Dataset, DatasetError
from .utils import resample_quaternion_array, quaternion_array_interpolate, quaternion_slerp, landmarks_from_sfm, \
                   position_from_sfm, orientation_from_gyro, orientation_from_sfm


class DatasetBuilder(object):
    LANDMARK_SOURCES = ('sfm', )
    SOURCES = ('imu', ) + LANDMARK_SOURCES

    def __init__(self):
        self._sfm = None
        self._gyro_data = None
        self._gyro_times = None

        self._orientation_source = None
        self._position_source = None
        self._landmark_source = None

    @property
    def selected_sources(self):
        return {
            'orientation' : self._orientation_source,
            'position' : self._position_source,
            'landmark' : self._landmark_source
        }

    def add_source_sfm(self, sfm):
        if self._sfm is None:
            self._sfm = sfm
        else:
            raise DatasetError("There is already an SfM source added")


    def add_source_gyro(self, gyro_data, gyro_times):
        n, d = gyro_data.shape
        if not n == len(gyro_times):
            raise DatasetError("Gyro data and timestamps length did not match")
        if not d == 3:
            raise DatasetError("Gyro data must have shape Nx3")

        if self._gyro_data is None:
            self._gyro_data = gyro_data
            self._gyro_times = gyro_times
            dt = float(gyro_times[1] - gyro_times[0])
            if not np.allclose(np.diff(gyro_times), dt):
                raise DatasetError("Gyro samples must be uniformly sampled")
            q = crisp.fastintegrate.integrate_gyro_quaternion_uniform(gyro_data, dt)
            self._gyro_quat = QuaternionArray(q)
        else:
            raise DatasetError("Can not add multiple gyro sources")

    def set_orientation_source(self, source):
        if source in self.SOURCES:
            self._orientation_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def set_position_source(self, source):
        if source in self.SOURCES:
            self._position_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def set_landmark_source(self, source):
        if source in self.LANDMARK_SOURCES:
            self._landmark_source = source
        else:
            raise DatasetError("No such source type: {}".format(source))

    def _can_build(self):
        return self._landmark_source is not None and \
                self._orientation_source is not None and \
                self._position_source is not None

    def _sfm_aligned_imu_orientations(self):
        view_times = np.array([v.time for v in self._sfm.views])
        view_idx = np.flatnonzero(view_times >= self._gyro_times[0])[0]
        view = self._sfm.views[view_idx]
        t_ref = view.time
        q_ref = view.orientation
        gstart_idx = np.argmin(np.abs(self._gyro_times - t_ref))
        q_initial = np.array(q_ref.components)
        gyro_part = self._gyro_data[gstart_idx:]
        gyro_part_times = self._gyro_times[gstart_idx:]
        dt = float(gyro_part_times[1] - gyro_part_times[0])
        gyro_part = gyro_part
        q = crisp.fastintegrate.integrate_gyro_quaternion_uniform(gyro_part, dt, initial=q_initial)
        return q, gyro_part_times


    def build(self):
        if not self._can_build():
            raise DatasetError("Must select all sources")
        ds = Dataset()
        ss = self.selected_sources

        if ss['landmark'] == 'sfm':
            view_times, landmarks = landmarks_from_sfm(self._sfm)
            ds.set_landmarks(view_times, landmarks)
        elif ss['landmark'] in self.LANDMARK_SOURCES:
            raise DatasetError("Loading landmarks from source '{}' is not yet implemented".format(ss['landmark']))
        else:
            raise DatasetError("Source type '{}' can not be used for landmarks".format(ss['landmark']))

        if ss['orientation'] == 'imu':
                orientations, timestamps = self._sfm_aligned_imu_orientations()
                timestamps, orientations = orientation_from_gyro(orientations, timestamps)
                ds.set_orientation_data(timestamps, orientations)
        elif ss['orientation'] == 'sfm':
            timestamps, orientations = orientation_from_sfm(self._sfm)
            ds.set_orientation_data(timestamps, orientations)
        else:
            raise DatasetError("'{}' source can not be used for orientations!".format(ss['orientation']))

        if ss['position'] == 'sfm':
            timestamps, positions = position_from_sfm(self._sfm)
            ds.set_position_data(timestamps, positions)
        else:
            raise DatasetError("'{}' source can not be used for position!".format(ss['position']))

        return ds
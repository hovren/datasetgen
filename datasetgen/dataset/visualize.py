from __future__ import print_function, division

import numpy as np
from mayavi import mlab

def visualize(ds, draw_orientations=False, draw_axes=False):

        t_min = ds.trajectory.startTime
        t_max = ds.trajectory.endTime
        t_samples = (t_max - t_min) * 50
        t = np.linspace(t_min, t_max, t_samples)
        positions = ds.trajectory.position(t)
        landmark_data = np.vstack([lm.position for lm in ds.landmarks]).T
        landmark_scalars = np.arange(len(ds.landmarks))

        landmark_colors = np.vstack([lm.color for lm in ds.landmarks])
        orientation_times = np.linspace(t_min, t_max, num=50)
        orientations = ds.trajectory.rotation(orientation_times)

        # World to camera transform is
        # Xc = RXw - Rt where R is the camera orientation and position respectively
        # Camera to world is thus
        # Xw = RtXc + t
        zc = np.array([0, 0, 1.]).reshape(3,1)
        zw = [np.dot(np.array(q.toMatrix()).T, zc).reshape(3,1) for q in orientations]
        quiver_pos = ds.trajectory.position(orientation_times)
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
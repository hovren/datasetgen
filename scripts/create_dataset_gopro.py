from __future__ import print_function
import os
import argparse
import time

from datasetgen.sfm import OpenMvgResult
from datasetgen.dataset import DatasetBuilder
from datasetgen.utilities.calibratedgyro import CalibratedGyroStream

parser = argparse.ArgumentParser()
parser.add_argument('videofile')
parser.add_argument("landmark_source")
parser.add_argument("output")
parser.add_argument("--no-gyro", action='store_true')
parser.add_argument("--no-color", action='store_true')
parser.add_argument("--no-align", action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--speed-scale', type=float, default=1.4)
args = parser.parse_args()

db = DatasetBuilder()

dataset_root, videofilename = os.path.split(args.videofile)
sequence = os.path.splitext(os.path.basename(videofilename))[0]
gyro_stream = CalibratedGyroStream.from_gopro(dataset_root, sequence)

db.add_source_gyro(gyro_stream.data, gyro_stream.timestamps)
if args.no_gyro:
    print('Using SfM for orientation')
    db.set_orientation_source('sfm')
else:
    db.set_orientation_source('imu')

color = not args.no_color
sfm = OpenMvgResult.from_file(args.landmark_source, camera_fps=30.0, color=color)

db.add_source_sfm(sfm)
db.set_landmark_source('sfm')
db.set_position_source('sfm')
align_knots = not args.no_align
print('Aligning knots:', align_knots)
ds = db.build(align_knots=align_knots)

ds = ds.rescaled_avg_speed(args.speed_scale)

print('Saving dataset to', args.output)
t0 = time.time()
ds.save(args.output, sequence)
elapsed = time.time() - t0
print('Saved {:d} landmarks in {:.1f} seconds'.format(len(ds.landmarks), elapsed))

if args.visualize:
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    t = np.linspace(ds.trajectory.startTime, ds.trajectory.endTime, num=2000)
    p = ds.trajectory.position(t)
    ang_vel = ds.trajectory.rotation(t).rotateFrame(ds.trajectory.rotationalVelocity(t))
    acc = ds.trajectory.acceleration(t)
    plt.subplot(3,1,1)
    plt.plot(t, p.T)
    plt.title('Position')
    plt.subplot(3,1,2)
    plt.plot(t, ang_vel.T)
    plt.title('Angular velocity')
    plt.subplot(3,1,3)
    plt.plot(t, acc.T)
    plt.title('Acceleration')

    plt.figure()
    x, y, z = p
    landmarks = np.hstack([lm.position.reshape(3,1) for lm in ds.landmarks])
    plt.plot(x, y, color='r', linewidth=4)
    plt.scatter(landmarks[0], landmarks[1], marker='x')
    plt.show()
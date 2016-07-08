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
ds = db.build()

ds = ds.rescaled_avg_speed(1.4)

print('Saving dataset to', args.output)
t0 = time.time()
ds.save(args.output, sequence)
elapsed = time.time() - t0
print('Saved {:d} landmarks in {:.1f} seconds'.format(len(ds.landmarks), elapsed))
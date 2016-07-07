from __future__ import print_function, division

import argparse
import os

import crisp

OPENMVG_BIN_PATH = '/home/hannes/Source/openMVG/build/Linux-x86_64-RELEASE/'

parser = argparse.ArgumentParser()
parser.add_argument('framedir')
parser.add_argument('camera')
parser.add_argument('sfmdir', default='./openmvg_tmp/')

def run_openmvg(tool, arg_str):
    program = os.path.join(OPENMVG_BIN_PATH, 'openMVG_main_{}'.format(tool))
    command = '{} {}'.format(program, arg_str)
    print('Running :', command)
    return os.system(command)

def openmvg_list_files(sfmdir, framedir, camera):
    K_str = ";".join([str(x) for x in camera.camera_matrix.ravel()])
    if not os.path.exists(os.path.join(sfmdir, 'sfm_data.json')):
        run_openmvg('SfMInit_ImageListing', '-k "{K}" -i {framedir} -o {sfmdir}'.format(sfmdir=sfmdir, framedir=framedir, K=K_str))

def openmvg_compute_features(sfmdir, framedir, preset='NORMAL'):
    run_openmvg('ComputeFeatures', '-o {framedir} -i {sfmdir}/sfm_data.json -m SIFT --describerPreset {preset}'.format(sfmdir=sfmdir,framedir=framedir, preset=preset))

def openmvg_compute_matches(sfmdir, framedir):
    run_openmvg('ComputeMatches', '-i {dir}/sfm_data.json -o {framedir} -r 0.8 -g e -f 1'.format(dir=sfmdir, framedir=framedir))

def openmvg_global_sfm(sfmdir, framedir=None):
    run_openmvg('GlobalSfM', '-i {sfmdir}/sfm_data.json  -m {sfmdir} -o {sfmdir}'.format(sfmdir=sfmdir, framedir=framedir))

args = parser.parse_args()

camera = crisp.AtanCameraModel.from_hdf(args.camera)
print(camera)

if not os.path.exists(args.sfmdir):
    print('Creating', args.sfmdir)
    os.makedirs(args.sfmdir)

if not os.path.exists(os.path.join(args.sfmdir, 'sfm_data.json')):
    openmvg_list_files(args.sfmdir, args.framedir, camera)

openmvg_compute_features(args.sfmdir, args.framedir)
openmvg_compute_matches(args.sfmdir, args.framedir)
openmvg_global_sfm(args.sfmdir, args.framedir)



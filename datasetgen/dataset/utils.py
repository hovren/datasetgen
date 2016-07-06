from __future__ import print_function, division

import numpy as np
import crisp.rotations

from ..maths.quaternions import Quaternion, QuaternionArray

def quaternion_slerp(q0, q1, tau):
    q0_arr = np.array([q0.w, q0.x, q0.y, q0.z])
    q1_arr = np.array([q1.w, q1.x, q1.y, q1.z])
    q_arr = crisp.rotations.slerp(q0_arr, q1_arr, tau)
    return Quaternion(*q_arr)


def quaternion_array_interpolate(qa, qtimes, t):
    i = np.flatnonzero(qtimes > t)[0]
    q0 = qa[i-1]
    q1 = qa[i]
    t0 = qtimes[i-1]
    t1 = qtimes[i]
    tau = np.clip((t - t0) / (t1 - t0), 0, 1)

    return quaternion_slerp(q0, q1, tau)


def resample_quaternion_array(qa, timestamps, resize=None):
    num_samples = resize if resize is not None else len(qa)
    timestamps_new = np.linspace(timestamps[0], timestamps[-1], num_samples)
    new_q = []
    unpack = lambda q: np.array([q.w, q.x, q.y, q.z])
    for t in timestamps_new:
        i = np.flatnonzero(timestamps >= t)[0]
        t1 = timestamps[i]
        if np.isclose(t1, t):
            new_q.append(qa[i])
        else:
            t0 = timestamps[i-1]
            tau = (t - t0) / (t1 - t0)
            q0 = qa[i-1]
            q1 = qa[i]
            qc = crisp.rotations.slerp(unpack(q0), unpack(q1), tau)
            q = Quaternion(qc[0], qc[1], qc[2], qc[3])
            new_q.append(q)
    return QuaternionArray(new_q), timestamps_new


def create_bounds(times):
    bounds = [-float('inf')] #[times[0] - 0.5*(times[1] - times[0])]
    for i in range(1, len(times)):
        ta = times[i-1]
        tb = times[i]
        bounds.append(0.5 * (ta + tb))
    #bounds.append(times[-1] + 0.5*(times[-1] - times[-2]))
    bounds.append(float('inf'))
    return bounds
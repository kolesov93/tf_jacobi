#!/usr/bin/env python3
import sys
import time

import numpy as np
import tensorflow as tf

def get_mask(N):
    return tf.pad(tf.ones((N - 1, N - 1)), ((1, 1), (1, 1)))

def step(u, f, h, mask):
    # Args:
    # u - [N + 1; N + 1] matrix (including borders)
    # f - [N - 1; N - 1] matrix (excluding borders)
    # h - grid size (1 / N)
    # mask - [N + 1, N + 1] matrix computed with get_mask(N)
    # Returns:
    # [N + 1; N + 1] matrix: result of the iteration

    up = tf.roll(u, shift=-1, axis=0)
    down = tf.roll(u, shift=1, axis=0)
    left = tf.roll(u, shift=-1, axis=1)
    right = tf.roll(u, shift=1, axis=1)
    update = ((left + right + up + down) - h*h*f) / 4.

    return update * mask + u * (1 - mask)

def constant_one(x, y):
    return 1.

def constant_zero(x, y):
    return 0.

def sample_heat_source(x, y):
    return -np.exp(-10. * ((x - 0.5)**2 + (y - 0.5)**2))

def solve(N, border_condition, heat_source, eps, device='/cpu:0'):
    # computation graph
    with tf.device(device):
        h = 1. / N
        u0 = np.zeros((N + 1, N + 1), dtype=np.float32)
        f = np.zeros((N + 1, N + 1), dtype=np.float32)

        for ix in range(N + 1):
            x = h * ix
            for iy in range(N + 1):
                y = h * iy
                if ix == 0 or iy == 0 or ix == N or iy == N:
                    u0[ix][iy] = border_condition(x, y)
                else:
                    f[ix][iy] = heat_source(x, y)

        u = tf.Variable(u0)
        mask = get_mask(N)

        # this is an operation to get next value of u, not the actual next value of u
        nextu = step(u, f, h, mask)
        # this is an operation to get max difference, not the actual value of max difference
        diff = tf.reduce_max(tf.abs(u - nextu))
        # this is an assignment operation
        finish_step = tf.assign(u, nextu)

        init_op = tf.initialize_all_variables()

    # actual computation
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        print('Starting evaluation', file=sys.stderr)
        start_time = time.time()
        session.run(init_op)

        iteration_num = 0
        while True:
            iteration_num += 1
            diff_value = session.run(diff)
            session.run(finish_step)

            if iteration_num % 1000 == 0:
                print('Iteration {}: {}'.format(iteration_num, diff_value), file=sys.stderr)
            if diff_value < eps:
                break
        print('Finished with {} iterations. Computation took {:.1f}s'.format(iteration_num, time.time() - start_time))

solve(1000, constant_one, constant_zero, 1e-6)
#solve(200, constant_zero, sample_heat_source, 1e-7)

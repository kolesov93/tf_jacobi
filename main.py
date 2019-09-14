#!/usr/bin/env python3
import sys
import time

import numpy as np
import tensorflow as tf


def build_iteration(u, f, h):
    """
    Build computation step of Jacobi method for a heat equation.

    We have current approximation in u variable and return the next step approximation.
    The solution is
    result[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h*h*f[i][j]) / 4

    In order to get the best of vectorized computation we aim to use matrix operations when
    available

    The idea is to have u-variable have additional columns and rows for borders. That allows us
    to add views (submatrices) of the u. E.g. if N = 4 we have (u - actual values, b - border):

    b|bbb|b
    -+---+-
    b|uuu|b
    b|uuu|b
    b|uuu|b
    -+---+-
    b|bbb|b

    Then we create four views (top, bottom, left, right):

    .|ttt|.        .|...|.        .|...|.        .|...|.
    -+---+-        -+---+-        -+---+-        -+---+-
    .|ttt|.        .|...|.        l|ll.|.        .|.rr|r
    .|ttt|.        .|bbb|.        l|ll.|.        .|.rr|r
    .|...|.        .|bbb|.        l|ll.|.        .|.rr|r
    -+---+-        -+---+-        -+---+-        -+---+-
    .|...|.        .|bbb|.        .|...|.        .|...|.

    This way the (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) becomes
    top + bottom + left + right

    In order to get N+1 x N+1 matrix as in input, we pad all top/bottom/left/right
    with zeros. In order not to rewrite borders we mask inner values with 1, outer
    with zeros, like:

    0|000|0
    -+---+-
    0|111|0
    0|111|0
    0|111|0
    -+---+-
    0|000|0

    Args:
        u - [N + 1; N + 1] tf.Variable (including borders)
        f - [N + 1; N + 1] np.array (including borders)
        h - grid step (1 / N)
        mask - [N + 1, N + 1] matrix computed with get_mask(N)
    Returns:
        [N + 1; N + 1] matrix: result of the iteration
    """

    N = f.shape[0] - 1
    mask = tf.pad(tf.ones((N - 1, N - 1)), ((1, 1), (1, 1)))

    top = u[0:-2, 1:-1]
    bottom = u[2:, 1:-1]
    left = u[1:-1, 0:-2]
    right = u[1:-1, 2:]

    update = (tf.pad(left + right + top + bottom, ((1,1), (1,1))) - h*h*f) / 4.

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

        # this is an operation to get next value of u, not the actual next value of u
        nextu = build_iteration(u, f, h)
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

solve(1000, constant_one, constant_zero, 2e-5, device='/cpu:0')
#solve(1000, constant_one, constant_zero, 2e-5, device='/gpu:0')
#solve(200, constant_zero, sample_heat_source, 1e-7)

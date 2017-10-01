import os
import copy
import math
import six
import numpy as np
from scipy.special import comb

def deco_print(line, end='\n'):
	six.print_('>==================> ' + line, end=end)

def deco_print_dict(dic):
	for key, value in dic.items():
		deco_print('{} : {}'.format(key, value))

def num_poly_feature(n, order=1, include_bias=False):
	num = int(include_bias)
	for o in range(1, order+1):
		num += comb(n+o-1, o, exact=True)
	return num

def decide_boundary(mean, std, x_left, x_right, factor=1.0):
	if x_left == '':
		x_left = math.floor((mean - 3 * std) * factor / 10) * 10
	else:
		x_left = float(x_left)
	if x_right == '':
		x_right = math.floor((mean + 3 * std) * factor / 10) * 10
	else:
		x_right = float(x_right)
	return x_left, x_right

def construct_nonlinear_function(sess, model, x_freeze, idx_output, idx_x, idx_y=None, idx_z=None, factor_x=1.0, factor_y=1.0, factor_z=1.0):
	x_input = copy.deepcopy(x_freeze)
	if idx_y is None:
		def f(x):
			x_input[idx_x] = x / factor_x
			prob, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_input[np.newaxis,:]})
			return prob[0][idx_output]
	elif idx_z is None:
		def f(x, y):
			x_input[idx_x] = x / factor_x
			x_input[idx_y] = y / factor_y
			prob, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_input[np.newaxis,:]})
			return prob[0][idx_output]
	else:
		def f(x, y, z):
			x_input[idx_x] = x / factor_x
			x_input[idx_y] = y / factor_y
			x_input[idx_z] = z / factor_z
			prob, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_input[np.newaxis,:]})
			return prob[0][idx_output]
	return f

def feature_ranking(logdir, idx2covariate, num=30, status_in=0, status_out=1, float_feature_only=False):
	gradient = np.load(os.path.join(logdir, 'ave_absolute_gradient.npy'))
	gradient = gradient[status_in, status_out]
	gradient_sorted = sorted([(i, gradient[i]) for i in range(len(gradient))], key=lambda t:-t[1])
	gradient_sorted = [(idx, idx2covariate[idx], grad) for idx, grad in gradient_sorted]
	if float_feature_only:
		gradient_sorted = [item for item in gradient_sorted if item[0] >= 237]
	return gradient_sorted[:num]

def feature_ranking_pair(logdir, idx2covariate, idx2pair, num=30, status_in=0, status_out=1):
	gradient = np.load(os.path.join(logdir, 'ave_absolute_gradient_2.npy'))
	gradient = gradient[status_in, status_out]
	gradient_sorted = sorted([(i, gradient[i]) for i in range(len(gradient))], key=lambda t:-t[1])
	gradient_sorted = [(idx2pair[idx], grad) for idx, grad in gradient_sorted]
	gradient_sorted = [(pair, (idx2covariate[pair[0]],idx2covariate[pair[1]]), grad) for pair, grad in gradient_sorted]
	return gradient_sorted[:num]

def feature_ranking_trio(logdir, idx2covariate, idx2trio, num=30, status_in=0, status_out=1):
	gradient = np.load(os.path.join(logdir, 'ave_absolute_gradient_3.npy'))
	gradient = gradient[status_in, status_out]
	gradient_sorted = sorted([(i, gradient[i]) for i in range(len(gradient))], key=lambda t:-t[1])
	gradient_sorted = [(idx2trio[idx], grad) for idx, grad in gradient_sorted]
	gradient_sorted = [(trio, (idx2covariate[trio[0]],idx2covariate[trio[1]],idx2covariate[trio[2]]), grad) for trio, grad in gradient_sorted]
	return gradient_sorted[:num]
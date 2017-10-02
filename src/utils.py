import os
import copy
import math
import six
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

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

def decide_boundary(mean, std, x_left, x_right, lower_bound, upper_bound, factor=1.0):
	if x_left == '':
		x_left = math.floor(max(mean - 3 * std, lower_bound) * factor / 10) * 10
	else:
		x_left = float(x_left)
	if x_right == '':
		x_right = math.ceil(min(mean + 3 * std, upper_bound) * factor / 10) * 10
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

def combine_two_plots(logdir, dl, idx, inIdx, outIdx):
	x = np.load(os.path.join(logdir, 'x_%d_inIdx_%s_outIdx_%d_%s.npz' %(idx, inIdx, outIdx, 'neural')))['x']
	y_neural = np.load(os.path.join(logdir, 'x_%d_inIdx_%s_outIdx_%d_%s.npz' %(idx, inIdx, outIdx, 'neural')))['y']
	y_logistic = np.load(os.path.join(logdir, 'x_%d_inIdx_%s_outIdx_%d_%s.npz' %(idx, inIdx, outIdx, 'logistic')))['y']
	plt.figure()
	plt.scatter(x, y_neural, color='b', label='neural network')
	plt.scatter(x, y_logistic, color='r', label='logistic regression')
	plt.xlabel(dl._idx2covariate[idx])
	plt.ylabel('Probability of Transition to %s' %dl._idx2outcome[outIdx])
	plt.legend()
	plt.savefig(os.path.join(logdir, 'x_%d_y_%d.pdf' %(idx, outIdx)))
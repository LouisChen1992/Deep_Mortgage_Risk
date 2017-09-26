import copy
import math
import numpy as np

def covariate_ranking_by_ave_absolute_gradient(idx2covariate, ave_absolute_gradient, state=(0,1)):
	gradient = ave_absolute_gradient[state[0]][state[1]]
	gradient_sort = sorted([(i,gradient[i]) for i in range(len(gradient))], key=lambda t:-t[1])
	return [(idx2covariate[i], grad) for (i, grad) in gradient_sort]

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
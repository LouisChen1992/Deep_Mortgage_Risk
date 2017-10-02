import os
import math
import copy
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from src.utils import deco_print, deco_print_dict, decide_boundary, construct_nonlinear_function
from src.model import Config, Model
from src.data_layer import DataInRamInputLayer

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('model', 'neural', 'neural/logistic')
tf.flags.DEFINE_string('task', '', 'Task: 1d_nonlinear/2d_nonlinear/2d_contour/3d_contour/3d_contour_slice')
tf.flags.DEFINE_string('plot_out', '', 'Path to save plots')
FLAGS = tf.flags.FLAGS

### Create Data Layer
deco_print('Creating Data Layer')
path = '/vol/Numpy_data_subprime_Test_new'
mode = 'analysis'
dl = DataInRamInputLayer(path=path, shuffle=False, load_file_list=False)
deco_print('Data Layer Created')
###

### Create Model
deco_print('Creating Model')
if FLAGS.model == 'neural':
	config = Config(feature_dim=291, num_category=7, batch_size=1, dropout=1.0)
elif FLAGS.model == 'logistic':
	config = Config(feature_dim=291, num_category=7, hidden_dim=[], batch_size=1, dropout=1.0)
model = Model(config, is_training=False)
deco_print('Read Following Config')
deco_print_dict(vars(config))
deco_print('Model Created')
###

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=50)
if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
	saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
	deco_print('Restored Checkpoint')
else:
	sess.run(tf.global_variables_initializer())
	deco_print('Random Initialization')

### Load X_stat_data
data = np.load(os.path.join(FLAGS.logdir, 'X_stat_Test.npz'))
mean = data['mean']
std = data['std']
###

if FLAGS.task == '1d_nonlinear':
	idx = int(input('Enter Variate Idx (237 - 290): '))
	idx_input = input('Enter Input Idx: (0 - 4): ')
	idx_output = int(input('Enter Output Idx: (0 - 7): '))
	factor = float(input('Enter Amplification Factor: '))
	x_idx_left = input('Enter Variate Lower Bound (default: mean - 3 * std): ')
	x_idx_right = input('Enter Variate Upper Bound (default: mean + 3 * std): ')
	x_idx_left, x_idx_right = decide_boundary(mean[idx], std[idx], x_idx_left, x_idx_right, factor)
	### construct nonlinear function
	if idx_input != '':
		for i in range(5):
			if i == int(idx_input):
				mean[i] = 1
			else:
				mean[i] = 0
	f = construct_nonlinear_function(sess, model, mean, idx_output, idx_x=idx, factor_x=factor)
	###
	x = np.linspace(x_idx_left, x_idx_right, 51)
	y = np.zeros(len(x))
	for i in range(len(x)):
		y[i] = f(x[i])

	plt.scatter(x, y)
	plt.xlabel(dl._idx2covariate[idx])
	plt.ylabel('Probability of Transition to %s' %dl._idx2outcome[idx_output])
	plt.savefig(os.path.join(FLAGS.plot_out, 'x_%d_y_%d_%s.pdf' %(idx, idx_output, FLAGS.model)))
elif FLAGS.task == '2d_nonlinear' or FLAGS.task == '2d_contour':
	idx_x = int(input('Enter Variate Idx For x (237 - 290): '))
	idx_y = int(input('Enter Variate Idx For y (237 - 290): '))
	idx_input = input('Enter Input Idx: (0 - 4): ')
	idx_output = int(input('Enter Output Idx: (0 - 7): '))
	factor_x = float(input('Enter Amplification Factor For x: '))
	factor_y = float(input('Enter Amplification Factor For y: '))
	x_idx_left = input('Enter Variate Lower Bound For x (default: mean - 3 * std): ')
	x_idx_right = input('Enter Variate Upper Bound For x (default: mean + 3 * std): ')
	y_idx_left = input('Enter Variate Lower Bound For y (default: mean - 3 * std): ')
	y_idx_right = input('Enter Variate Upper Bound For y (default: mean + 3 * std): ')
	x_idx_left, x_idx_right = decide_boundary(mean[idx_x], std[idx_x], x_idx_left, x_idx_right, factor_x)
	y_idx_left, y_idx_right = decide_boundary(mean[idx_y], std[idx_y], y_idx_left, y_idx_right, factor_y)
	### construct nonlinear function
	if idx_input != '':
		for i in range(5):
			if i == int(idx_input):
				mean[i] = 1
			else:
				mean[i] = 0
	f = construct_nonlinear_function(sess, model, mean, idx_output, idx_x=idx_x, idx_y=idx_y, factor_x=factor_x, factor_y=factor_y)
	###
	x = np.linspace(x_idx_left, x_idx_right, 51)
	y = np.linspace(y_idx_left, y_idx_right, 51)
	z = np.zeros((len(y), len(x)))
	x, y = np.meshgrid(x, y)
	for i in range(len(y)):
		for j in range(len(x)):
			z[i,j] = f(x[i,j], y[i,j])
	if FLAGS.task == '2d_nonlinear':
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		ax.set_zlim(0, np.max(z))
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		# fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel(dl._idx2covariate[idx_x])
		ax.set_ylabel(dl._idx2covariate[idx_y])
		ax.set_zlabel('Probability of Transition to %s' %dl._idx2outcome[idx_output])
	elif FLAGS.task == '2d_contour':
		im = plt.contourf(x,y,z)
		plt.xlabel(dl._idx2covariate[idx_x])
		plt.ylabel(dl._idx2covariate[idx_y])
		cbar = plt.colorbar(im)
		cbar.ax.set_ylabel('Probability of Transition to %s' %dl._idx2outcome[idx_output])
	plt.savefig(os.path.join(FLAGS.plot_out, 'x_%d_y_%d_z_%d_%s.pdf' %(idx_x, idx_y, idx_output, FLAGS.model)))
elif FLAGS.task == '3d_contour':
	idx_x = int(input('Enter Variate Idx For x (237 - 290): '))
	idx_y = int(input('Enter Variate Idx For y (237 - 290): '))
	idx_z = int(input('Enter Variate Idx For z (237 - 290): '))
	idx_input = input('Enter Input Idx: (0 - 4): ')
	idx_output = int(input('Enter Output Idx: (0 - 7): '))
	factor_x = float(input('Enter Amplification Factor For x: '))
	factor_y = float(input('Enter Amplification Factor For y: '))
	factor_z = float(input('Enter Amplification Factor For z: '))
	x_idx_left = input('Enter Variate Lower Bound For x (default: mean - 3 * std): ')
	x_idx_right = input('Enter Variate Upper Bound For x (default: mean + 3 * std): ')
	y_idx_left = input('Enter Variate Lower Bound For y (default: mean - 3 * std): ')
	y_idx_right = input('Enter Variate Upper Bound For y (default: mean + 3 * std): ')
	z_idx_left = input('Enter Variate Lower Bound For z (default: mean - 3 * std): ')
	z_idx_right = input('Enter Variate Upper Bound For z (default: mean + 3 * std): ')
	x_idx_left, x_idx_right = decide_boundary(mean[idx_x], std[idx_x], x_idx_left, x_idx_right, factor_x)
	y_idx_left, y_idx_right = decide_boundary(mean[idx_y], std[idx_y], y_idx_left, y_idx_right, factor_y)
	z_idx_left, z_idx_right = decide_boundary(mean[idx_z], std[idx_z], z_idx_left, z_idx_right, factor_z)
	### construct nonlinear function
	if idx_input != '':
		for i in range(5):
			if i == int(idx_input):
				mean[i] = 1
			else:
				mean[i] = 0
	f = construct_nonlinear_function(sess, model, mean, idx_output, idx_x=idx_x, idx_y=idx_y, idx_z=idx_z, factor_x=factor_x, factor_y=factor_y, factor_z=factor_z)
	###
	x, y, z = np.mgrid[x_idx_left:x_idx_right:11j, y_idx_left:y_idx_right:11j, z_idx_left:z_idx_right:11j]
	v = np.zeros((11, 11, 11))
	for i in range(11):
		for j in range(11):
			for k in range(11):
				v[i,j,k] = f(x[i,j,k], y[i,j,k], z[i,j,k])
	### import mayavi
	from mayavi import mlab
	###
	mlab.contour3d(x, y, z, v, contours=10, extent=[0,1,0,1,0,1], opacity=0.5)
	mlab.outline()
	mlab.colorbar(orientation='vertical')
	# mlab.axes(ranges=[x_idx_left, x_idx_right, y_idx_left, y_idx_right, z_idx_left, z_idx_right], xlabel=dl._idx2covariate[idx_x], ylabel=dl._idx2covariate[idx_y], zlabel=dl._idx2covariate[idx_z])
	mlab.axes(ranges=[x_idx_left, x_idx_right, y_idx_left, y_idx_right, z_idx_left, z_idx_right])
	mlab.show()
elif FLAGS.task == '3d_contour_slice':
	idx_x = int(input('Enter Variate Idx For x (237 - 290): '))
	idx_y = int(input('Enter Variate Idx For y (237 - 290): '))
	idx_z = int(input('Enter Variate Idx For z (237 - 290): '))
	idx_input = input('Enter Input Idx: (0 - 4): ')
	idx_output = int(input('Enter Output Idx: (0 - 7): '))
	factor_x = float(input('Enter Amplification Factor For x: '))
	factor_y = float(input('Enter Amplification Factor For y: '))
	factor_z = float(input('Enter Amplification Factor For z: '))
	x_idx_left = input('Enter Variate Lower Bound For x (default: mean - 3 * std): ')
	x_idx_right = input('Enter Variate Upper Bound For x (default: mean + 3 * std): ')
	y_idx_left = input('Enter Variate Lower Bound For y (default: mean - 3 * std): ')
	y_idx_right = input('Enter Variate Upper Bound For y (default: mean + 3 * std): ')
	z_idx_left = input('Enter Variate Lower Bound For z (default: mean - 3 * std): ')
	z_idx_right = input('Enter Variate Upper Bound For z (default: mean + 3 * std): ')
	x_idx_left, x_idx_right = decide_boundary(mean[idx_x], std[idx_x], x_idx_left, x_idx_right, factor_x)
	y_idx_left, y_idx_right = decide_boundary(mean[idx_y], std[idx_y], y_idx_left, y_idx_right, factor_y)
	z_idx_left, z_idx_right = decide_boundary(mean[idx_z], std[idx_z], z_idx_left, z_idx_right, factor_z)
	x = np.linspace(x_idx_left, x_idx_right, 51)
	y = np.linspace(y_idx_left, y_idx_right, 51)
	zs = np.linspace(z_idx_left, z_idx_right, 4)
	v = np.zeros((len(y), len(x), len(zs)))
	x, y = np.meshgrid(x, y)
	### 3d contour slice
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	###
	if idx_input != '':
		for i in range(5):
			if i == int(idx_input):
				mean[i] = 1
			else:
				mean[i] = 0
	mean_copy = copy.deepcopy(mean)
	for k in range(len(zs)):
		z = zs[k]
		mean_copy[idx_z] = z / factor_z
		f = construct_nonlinear_function(sess, model, mean_copy, idx_output, idx_x=idx_x, idx_y=idx_y, factor_x=factor_x, factor_y=factor_y)
		for i in range(len(y)):
			for j in range(len(x)):
				v[i,j,k] = f(x[i,j], y[i,j])
	levels = np.linspace(np.min(v),np.max(v),10)
	for k in range(len(zs)):
		z = zs[k]
		im = ax.contourf(x, y, v[:,:,k], offset=z, levels=levels)
	ax.set_xlabel(dl._idx2covariate[idx_x])
	ax.set_ylabel(dl._idx2covariate[idx_y])
	ax.set_zlabel(dl._idx2covariate[idx_z])
	ax.set_xlim(x_idx_left, x_idx_right)
	ax.set_ylim(y_idx_left, y_idx_right)
	ax.set_zlim(z_idx_left, z_idx_right)
	cbar = plt.colorbar(im)
	cbar.ax.set_ylabel('Probability of Transition to %s' %dl._idx2outcome[idx_output])
	plt.show()
else:
	raise ValueError('Task Not Supported! ')
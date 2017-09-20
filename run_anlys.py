import os
import math
import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import deco_print, deco_print_dict
from src.utils_anlys import covariate_ranking_by_ave_absolute_gradient
from src.model import Config, Model
from src.data_layer import DataInRamInputLayer

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('task', '', 'Task: cov_rank/1d_nonlinear')
tf.flags.DEFINE_string('plot_out', '', 'Path to save plots')
FLAGS = tf.flags.FLAGS

### Create Data Layer
deco_print('Creating Data Layer')
path = '/vol/Numpy_data_subprime_Test_new'
mode = 'analysis'
dl = DataInRamInputLayer(path=path, mode=mode)
deco_print('Data Layer Created')
###

if FLAGS.task == 'cov_rank':
	ave_absolute_gradient = np.load(os.path.join(FLAGS.logdir, 'ave_absolute_gradient.npy'))
	ave_absolute_gradient_sort = covariate_ranking_by_ave_absolute_gradient(dl._idx2covariate, ave_absolute_gradient)
	for item in ave_absolute_gradient_sort:
		print(item)
else:
	### Create Model
	deco_print('Creating Model')
	config = Config(feature_dim=291, num_category=7, batch_size=1, dropout=1.0)
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

	if FLAGS.task == '1d_nonlinear':
		idx = int(input('Enter Variate Idx (237 - 290): '))
		factor = float(input('Enter Amplification Factor: '))
		idx_output = int(input('Enter Output Idx: (0 - 7)'))
		data = np.load(os.path.join(FLAGS.logdir, 'X_stat_Test.npz'))
		mean = data['mean']
		std = data['std']
		x_idx_left = math.floor((mean[idx] - 3 * std[idx]) * factor / 10) * 10
		x_idx_right = math.ceil((mean[idx] + 3 * std[idx]) * factor / 10) * 10
		x_idx = np.linspace(x_idx_left, x_idx_right, (x_idx_right - x_idx_left) // 10 + 1)

		def f(x):
			x_input = copy.deepcopy(mean)
			y = np.zeros(len(x))
			for i in range(len(x)):
				x_input[idx] = x[i] / factor
				prob, =sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_input[np.newaxis,:]})
				y[i] = prob[idx_output]
			return y

		y = f(x_idx)
		plt.scatter(x_idx, y)
		plt.xlabel(dl._idx2covariate[idx])
		plt.ylabel('Probability of Transition to %s' %dl._idx2outcome[idx_output])
		plt.savefig(os.path.join(FLAGS.plot_out, 'x_%d_y_%d.pdf' %(idx, idx_output)))

	else:
		raise ValueError('Task Not Supported! ')

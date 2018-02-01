import os
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from src.model import Config, Model
from src.data_layer import DataInRamInputLayer
from src.utils import deco_print, deco_print_dict

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_integer('num_epochs', 6000, 'Number of training epochs')
tf.flags.DEFINE_integer('leave_out_idx', -1, 'Index leave out covariate')
FLAGS = tf.flags.FLAGS

deco_print('Creating Data Layer')
path = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_new')
path_valid = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Val_new')
dl = DataInRamInputLayer(path=path, shuffle=True)
dl_valid = DataInRamInputLayer(path=path_valid, shuffle=True)
deco_print('Leave One Covariate out: %s' %dl._idx2covariate[FLAGS.leave_out_idx])
deco_print('Data Layer Created')

deco_print('Creating Model')
config = Config(feature_dim=291, num_category=7, dropout=0.9)
config_valid = Config(feature_dim=291, num_category=7, dropout=1.0)
with tf.variable_scope('nn_1'):
	model_1 = Model(config)
	model_1_valid = Model(config_valid, force_var_reuse=True, is_training=False)
with tf.variable_scope('nn_2'):
	model_2 = Model(config)
	model_2_valid = Model(config_valid, force_var_reuse=True, is_training=False)
deco_print('Read Following Config')
deco_print_dict(vars(config))
deco_print('Model Created')

with tf.Session() as sess:
	saver = tf.train.Saver(max_to_keep=50)
	if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
		deco_print('Restored Checkpoint')
	else:
		sess.run(tf.global_variables_initializer())
		deco_print('Random Initialization')

	deco_print('Executing Training Mode\n')
	summary_op = tf.summary.merge_all()
	sw = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

	for epoch in range(FLAGS.num_epochs):
		total_epoch_step_loss_1 = 0.0
		total_epoch_step_loss_2 = 0.0
		count_epoch_step = 0
		for i, (x, y, info) in enumerate(dl.iterate_one_epoch_step(config.batch_size)):
			feed_dict_1 = {model_1._x_placeholder:x, model_1._y_placeholder:y, model_1._epoch_step:info['epoch_step']}
			loss_1_i, _ = sess.run(fetches=[model_1._loss, model_1._train_op], feed_dict=feed_dict_1)
			x[:,FLAGS.leave_out_idx] = 0.0
			feed_dict_2 = {model_2._x_placeholder:x, model_2._y_placeholder:y, model_2._epoch_step:info['epoch_step']}
			loss_2_i, _ = sess.run(fetches=[model_2._loss, model_2._train_op], feed_dict=feed_dict_2)
			total_epoch_step_loss_1 += loss_1_i
			total_epoch_step_loss_2 += loss_2_i
			count_epoch_step += 1

		train_epoch_step_loss_1 = total_epoch_step_loss_1 / count_epoch_step
		train_epoch_step_loss_2 = total_epoch_step_loss_2 / count_epoch_step
		deco_print('Epoch {} Training Loss: {} AND {}'.format(epoch, train_epoch_step_loss_1, train_epoch_step_loss_2))
		train_loss_value_epoch_step_1 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_1', simple_value=train_epoch_step_loss_1)
		train_loss_value_epoch_step_2 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_2', simple_value=train_epoch_step_loss_2)
		summary = summary_pb2.Summary(value=[train_loss_value_epoch_step_1, train_loss_value_epoch_step_2])
		sw.add_summary(summary, global_step=epoch)
		sw.flush()

		total_valid_loss_1 = 0.0
		total_valid_loss_2 = 0.0
		count_valid = 0
		for i, (x, y, _) in enumerate(dl_valid.iterate_one_epoch_step(config_valid.batch_size)):
			feed_dict_1 = {model_1_valid._x_placeholder:x, model_1_valid._y_placeholder:y}
			loss_1_i, = sess.run(fetches=[model_1_valid._loss], feed_dict=feed_dict_1)
			x[:,FLAGS.leave_out_idx] = 0.0
			feed_dict_2 = {model_2_valid._x_placeholder:x, model_2_valid._y_placeholder:y}
			loss_2_i, = sess.run(fetches=[model_2_valid._loss], feed_dict=feed_dict_2)
			total_valid_loss_1 += loss_1_i
			total_valid_loss_2 += loss_2_i
			count_valid += 1

		valid_loss_1 = total_valid_loss_1 / count_valid
		valid_loss_2 = total_valid_loss_2 / count_valid
		deco_print('Epoch {} Validation Loss: {} AND {}'.format(epoch, valid_loss_1, valid_loss_2))
		valid_loss_value_1 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_1', simple_value=valid_loss_1)
		valid_loss_value_2 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_2', simple_value=valid_loss_2)
		summary = summary_pb2.Summary(value=[valid_loss_value_1, valid_loss_value_2])
		sw.add_summary(summary=summary, global_step=epoch)
		sw.flush()

	saver.save(sess, save_path=os.path.join(FLAGS.logdir, 'model'), global_step=FLAGS.num_epochs)
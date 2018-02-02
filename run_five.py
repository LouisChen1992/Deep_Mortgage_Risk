import os
import numpy as np
import copy
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from src.model import Config, Model
from src.data_layer import DataInRamInputLayer
from src.utils import deco_print, deco_print_dict

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_integer('num_epochs', 5000, 'Number of training epochs')
FLAGS = tf.flags.FLAGS

### 251: state unemployment rate
### 266: current outstanding balance
### 239: original interest rate
### 237: FICO score
deco_print('Creating Data Layer')
path = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_new')
path_valid = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Val_new')
path_test = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Test_new')
dl = DataInRamInputLayer(path=path, shuffle=True)
dl_valid = DataInRamInputLayer(path=path_valid, shuffle=True)
dl_test = DataInRamInputLayer(path=path_test, shuffle=False)
deco_print('Leave Covariates out: %s, %s, %s, %s' %(dl._idx2covariate[251],dl._idx2covariate[266], dl._idx2covariate[239], dl._idx2covariate[237]))
deco_print('Data Layer Created')

deco_print('Creating Model')
config = Config(feature_dim=291, num_category=7, dropout=0.9)
config_valid = Config(feature_dim=291, num_category=7, dropout=1.0)
with tf.variable_scope('nn'):
	model = Model(config)
	model_valid = Model(config_valid, force_var_reuse=True, is_training=False)
with tf.variable_scope('nn_251'):
	model_251 = Model(config)
	model_251_valid = Model(config_valid, force_var_reuse=True, is_training=False)
with tf.variable_scope('nn_266'):
	model_266 = Model(config)
	model_266_valid = Model(config_valid, force_var_reuse=True, is_training=False)
with tf.variable_scope('nn_239'):
	model_239 = Model(config)
	model_239_valid = Model(config_valid, force_var_reuse=True, is_training=False)
with tf.variable_scope('nn_237'):
	model_237 = Model(config)
	model_237_valid = Model(config_valid, force_var_reuse=True, is_training=False)
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
		total_epoch_step_loss = 0.0
		total_epoch_step_loss_251 = 0.0
		total_epoch_step_loss_266 = 0.0
		total_epoch_step_loss_239 = 0.0
		total_epoch_step_loss_237 = 0.0
		count_epoch_step = 0
		for i, (x, y, info) in enumerate(dl.iterate_one_epoch_step(config.batch_size)):
			feed_dict = {model._x_placeholder:x, model._y_placeholder:y, model._epoch_step:info['epoch_step']}
			loss_i, _ = sess.run(fetches=[model._loss, model._train_op], feed_dict=feed_dict)

			x_copy = copy.deepcopy(x)
			x_copy[:,251] = 0.0
			feed_dict_251 = {model_251._x_placeholder:x_copy, model_251._y_placeholder:y, model_251._epoch_step:info['epoch_step']}
			loss_251_i, _ = sess.run(fetches=[model_251._loss, model_251._train_op], feed_dict=feed_dict_251)

			x_copy = copy.deepcopy(x)
			x_copy[:,266] = 0.0
			feed_dict_266 = {model_266._x_placeholder:x_copy, model_266._y_placeholder:y, model_266._epoch_step:info['epoch_step']}
			loss_266_i, _ = sess.run(fetches=[model_266._loss, model_266._train_op], feed_dict=feed_dict_266)

			x_copy = copy.deepcopy(x)
			x_copy[:,239] = 0.0
			feed_dict_239 = {model_239._x_placeholder:x_copy, model_239._y_placeholder:y, model_239._epoch_step:info['epoch_step']}
			loss_239_i, _ = sess.run(fetches=[model_239._loss, model_239._train_op], feed_dict=feed_dict_239)

			x_copy = copy.deepcopy(x)
			x_copy[:,237] = 0.0
			feed_dict_237 = {model_237._x_placeholder:x_copy, model_237._y_placeholder:y, model_237._epoch_step:info['epoch_step']}
			loss_237_i, _ = sess.run(fetches=[model_237._loss, model_237._train_op], feed_dict=feed_dict_237)

			total_epoch_step_loss += loss_i
			total_epoch_step_loss_251 += loss_251_i
			total_epoch_step_loss_266 += loss_266_i
			total_epoch_step_loss_239 += loss_239_i
			total_epoch_step_loss_237 += loss_237_i
			count_epoch_step += 1

		train_epoch_step_loss = total_epoch_step_loss / count_epoch_step
		train_epoch_step_loss_251 = total_epoch_step_loss_251 / count_epoch_step
		train_epoch_step_loss_266 = total_epoch_step_loss_266 / count_epoch_step
		train_epoch_step_loss_239 = total_epoch_step_loss_239 / count_epoch_step
		train_epoch_step_loss_237 = total_epoch_step_loss_237 / count_epoch_step
		deco_print('Epoch %d Training Loss: %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f' %(epoch, train_epoch_step_loss, train_epoch_step_loss_251, train_epoch_step_loss_266, train_epoch_step_loss_239, train_epoch_step_loss_237), end='\r')
		train_loss_value_epoch_step = summary_pb2.Summary.Value(tag='epoch_step_train_loss', simple_value=train_epoch_step_loss)
		train_loss_value_epoch_step_251 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_251', simple_value=train_epoch_step_loss_251)
		train_loss_value_epoch_step_266 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_266', simple_value=train_epoch_step_loss_266)
		train_loss_value_epoch_step_239 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_239', simple_value=train_epoch_step_loss_239)
		train_loss_value_epoch_step_237 = summary_pb2.Summary.Value(tag='epoch_step_train_loss_237', simple_value=train_epoch_step_loss_237)
		summary = summary_pb2.Summary(value=[train_loss_value_epoch_step, train_loss_value_epoch_step_251, train_loss_value_epoch_step_266, train_loss_value_epoch_step_239, train_loss_value_epoch_step_237])
		sw.add_summary(summary, global_step=epoch)
		sw.flush()

		total_valid_loss = 0.0
		total_valid_loss_251 = 0.0
		total_valid_loss_266 = 0.0
		total_valid_loss_239 = 0.0
		total_valid_loss_237 = 0.0
		count_valid = 0
		for i, (x, y, _) in enumerate(dl_valid.iterate_one_epoch_step(config_valid.batch_size)):
			feed_dict = {model_valid._x_placeholder:x, model_valid._y_placeholder:y}
			loss_i, = sess.run(fetches=[model_valid._loss], feed_dict=feed_dict)

			x_copy = copy.deepcopy(x)
			x_copy[:,251] = 0.0
			feed_dict_251 = {model_251_valid._x_placeholder:x_copy, model_251_valid._y_placeholder:y}
			loss_251_i, = sess.run(fetches=[model_251_valid._loss], feed_dict=feed_dict_251)

			x_copy = copy.deepcopy(x)
			x_copy[:,266] = 0.0
			feed_dict_266 = {model_266_valid._x_placeholder:x_copy, model_266_valid._y_placeholder:y}
			loss_266_i, = sess.run(fetches=[model_266_valid._loss], feed_dict=feed_dict_266)

			x_copy = copy.deepcopy(x)
			x_copy[:,239] = 0.0
			feed_dict_239 = {model_239_valid._x_placeholder:x_copy, model_239_valid._y_placeholder:y}
			loss_239_i, = sess.run(fetches=[model_239_valid._loss], feed_dict=feed_dict_239)

			x_copy = copy.deepcopy(x)
			x_copy[:,237] = 0.0
			feed_dict_237 = {model_237_valid._x_placeholder:x_copy, model_237_valid._y_placeholder:y}
			loss_237_i, = sess.run(fetches=[model_237_valid._loss], feed_dict=feed_dict_237)

			total_valid_loss += loss_i
			total_valid_loss_251 += loss_251_i
			total_valid_loss_266 += loss_266_i
			total_valid_loss_239 += loss_239_i
			total_valid_loss_237 += loss_237_i
			count_valid += 1

		valid_loss = total_valid_loss / count_valid
		valid_loss_251 = total_valid_loss_251 / count_valid
		valid_loss_266 = total_valid_loss_266 / count_valid
		valid_loss_239 = total_valid_loss_239 / count_valid
		valid_loss_237 = total_valid_loss_237 / count_valid
		deco_print('Epoch %d Validation Loss: %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f' %(epoch, valid_loss, valid_loss_251, valid_loss_266, valid_loss_239, valid_loss_237), end='\r')
		valid_loss_value = summary_pb2.Summary.Value(tag='epoch_step_valid_loss', simple_value=valid_loss)
		valid_loss_value_251 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_251', simple_value=valid_loss_251)
		valid_loss_value_266 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_266', simple_value=valid_loss_266)
		valid_loss_value_239 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_239', simple_value=valid_loss_239)
		valid_loss_value_237 = summary_pb2.Summary.Value(tag='epoch_step_valid_loss_237', simple_value=valid_loss_237)
		summary = summary_pb2.Summary(value=[valid_loss_value, valid_loss_value_251, valid_loss_value_266, valid_loss_value_239, valid_loss_value_237])
		sw.add_summary(summary=summary, global_step=epoch)
		sw.flush()

	saver.save(sess, save_path=os.path.join(FLAGS.logdir, 'model'), global_step=FLAGS.num_epochs)
	deco_print('Training Finished! \n')

	total_train_loss = 0.0
	total_train_loss_251 = 0.0
	total_train_loss_266 = 0.0
	total_train_loss_239 = 0.0
	total_train_loss_237 = 0.0
	count_train = 0
	for i, (x, y, _) in enumerate(dl.iterate_one_epoch(config.batch_size)):
		feed_dict = {model_valid._x_placeholder:x, model_valid._y_placeholder:y}
		loss_i, = sess.run(fetches=[model_valid._loss], feed_dict=feed_dict)

		x_copy = copy.deepcopy(x)
		x_copy[:,251] = 0.0
		feed_dict_251 = {model_251_valid._x_placeholder:x_copy, model_251_valid._y_placeholder:y}
		loss_i_251, = sess.run(fetches=[model_251_valid._loss], feed_dict=feed_dict_251)

		x_copy = copy.deepcopy(x)
		x_copy[:,266] = 0.0
		feed_dict_266 = {model_266_valid._x_placeholder:x_copy, model_266_valid._y_placeholder:y}
		loss_i_266, = sess.run(fetches=[model_266_valid._loss], feed_dict=feed_dict_266)

		x_copy = copy.deepcopy(x)
		x_copy[:,239] = 0.0
		feed_dict_239 = {model_239_valid._x_placeholder:x_copy, model_239_valid._y_placeholder:y}
		loss_i_239, = sess.run(fetches=[model_239_valid._loss], feed_dict=feed_dict_239)

		x_copy = copy.deepcopy(x)
		x_copy[:,237] = 0.0
		feed_dict_237 = {model_237_valid._x_placeholder:x_copy, model_237_valid._y_placeholder:y}
		loss_i_237, = sess.run(fetches=[model_237_valid._loss], feed_dict=feed_dict_237)

		total_train_loss += loss_i
		total_train_loss_251 += loss_i_251
		total_train_loss_266 += loss_i_266
		total_train_loss_239 += loss_i_239
		total_train_loss_237 += loss_i_237
		count_train += 1

	deco_print('Training Loss: %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f' %(total_train_loss/count_train, total_train_loss_251/count_train, total_train_loss_266/count_train, total_train_loss_239/count_train, total_train_loss_237/count_train))

	total_valid_loss = 0.0
	total_valid_loss_251 = 0.0
	total_valid_loss_266 = 0.0
	total_valid_loss_239 = 0.0
	total_valid_loss_237 = 0.0
	count_valid = 0
	for i, (x, y, _) in enumerate(dl_valid.iterate_one_epoch(config.batch_size)):
		feed_dict = {model_valid._x_placeholder:x, model_valid._y_placeholder:y}
		loss_i, = sess.run(fetches=[model_valid._loss], feed_dict=feed_dict)

		x_copy = copy.deepcopy(x)
		x_copy[:,251] = 0.0
		feed_dict_251 = {model_251_valid._x_placeholder:x_copy, model_251_valid._y_placeholder:y}
		loss_i_251, = sess.run(fetches=[model_251_valid._loss], feed_dict=feed_dict_251)

		x_copy = copy.deepcopy(x)
		x_copy[:,266] = 0.0
		feed_dict_266 = {model_266_valid._x_placeholder:x_copy, model_266_valid._y_placeholder:y}
		loss_i_266, = sess.run(fetches=[model_266_valid._loss], feed_dict=feed_dict_266)

		x_copy = copy.deepcopy(x)
		x_copy[:,239] = 0.0
		feed_dict_239 = {model_239_valid._x_placeholder:x_copy, model_239_valid._y_placeholder:y}
		loss_i_239, = sess.run(fetches=[model_239_valid._loss], feed_dict=feed_dict_239)

		x_copy = copy.deepcopy(x)
		x_copy[:,237] = 0.0
		feed_dict_237 = {model_237_valid._x_placeholder:x_copy, model_237_valid._y_placeholder:y}
		loss_i_237, = sess.run(fetches=[model_237_valid._loss], feed_dict=feed_dict_237)

		total_valid_loss += loss_i
		total_valid_loss_251 += loss_i_251
		total_valid_loss_266 += loss_i_266
		total_valid_loss_239 += loss_i_239
		total_valid_loss_237 += loss_i_237
		count_valid += 1

	deco_print('Validation Loss: %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f' %(total_valid_loss/count_valid, total_valid_loss_251/count_valid, total_valid_loss_266/count_valid, total_valid_loss_239/count_valid, total_valid_loss_237/count_valid))

	total_test_loss = 0.0
	total_test_loss_251 = 0.0
	total_test_loss_266 = 0.0
	total_test_loss_239 = 0.0
	total_test_loss_237 = 0.0
	count_test = 0
	for i, (x, y, _) in enumerate(dl_test.iterate_one_epoch(config.batch_size)):
		feed_dict = {model_valid._x_placeholder:x, model_valid._y_placeholder:y}
		loss_i, = sess.run(fetches=[model_valid._loss], feed_dict=feed_dict)

		x_copy = copy.deepcopy(x)
		x_copy[:,251] = 0.0
		feed_dict_251 = {model_251_valid._x_placeholder:x_copy, model_251_valid._y_placeholder:y}
		loss_i_251, = sess.run(fetches=[model_251_valid._loss], feed_dict=feed_dict_251)

		x_copy = copy.deepcopy(x)
		x_copy[:,266] = 0.0
		feed_dict_266 = {model_266_valid._x_placeholder:x_copy, model_266_valid._y_placeholder:y}
		loss_i_266, = sess.run(fetches=[model_266_valid._loss], feed_dict=feed_dict_266)

		x_copy = copy.deepcopy(x)
		x_copy[:,239] = 0.0
		feed_dict_239 = {model_239_valid._x_placeholder:x_copy, model_239_valid._y_placeholder:y}
		loss_i_239, = sess.run(fetches=[model_239_valid._loss], feed_dict=feed_dict_239)

		x_copy = copy.deepcopy(x)
		x_copy[:,237] = 0.0
		feed_dict_237 = {model_237_valid._x_placeholder:x_copy, model_237_valid._y_placeholder:y}
		loss_i_237, = sess.run(fetches=[model_237_valid._loss], feed_dict=feed_dict_237)

		total_test_loss += loss_i
		total_test_loss_251 += loss_i_251
		total_test_loss_266 += loss_i_266
		total_test_loss_239 += loss_i_239
		total_test_loss_237 += loss_i_237
		count_test += 1

	deco_print('Test Loss: %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f' %(total_test_loss/count_test, total_test_loss_251/count_test, total_test_loss_266/count_test, total_test_loss_239/count_test, total_test_loss_237/count_test))





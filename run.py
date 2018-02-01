import os
import time
import copy
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from src.model import Config, Model
from src.data_layer import DataInRamInputLayer
from src.utils import deco_print, deco_print_dict, feature_ranking, feature_ranking_pair, feature_ranking_trio

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('mode', 'train', 'Mode: train/test/sens_anlys/sens_anlys_pair/sens_anlys_trio')
tf.flags.DEFINE_integer('sample_size', -100, 'Number of samples')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of training epochs')
tf.flags.DEFINE_float('delta', 1.1, 'Delta')
tf.flags.DEFINE_integer('leave_out_idx', -1, 'Index leave out covariate')
FLAGS = tf.flags.FLAGS

### Create Data Layer
deco_print('Creating Data Layer')
if FLAGS.mode == 'train':
	path = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_new')
	dl = DataInRamInputLayer(path=path, shuffle=True, leave_out_idx=FLAGS.leave_out_idx)
	path_valid = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Val_new')
	dl_valid = DataInRamInputLayer(path=path_valid, shuffle=False, leave_out_idx=FLAGS.leave_out_idx)
elif FLAGS.mode == 'test':
	path = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Test_new')
	dl = DataInRamInputLayer(path=path, shuffle=False, leave_out_idx=FLAGS.leave_out_idx)
elif FLAGS.mode == 'sens_anlys' or FLAGS.mode == 'sens_anlys_pair' or FLAGS.mode == 'sens_anlys_trio':
	path = os.path.join(os.path.expanduser('~'), 'data/vol/Numpy_data_subprime_Test_new')
	if FLAGS.sample_size == -100:
		dl = DataInRamInputLayer(path=path, shuffle=False, leave_out_idx=FLAGS.leave_out_idx)
	else:
		dl = DataInRamInputLayer(path=path, shuffle=True, leave_out_idx=FLAGS.leave_out_idx)
else:
	raise ValueError('Mode Not Implemented')
if FLAGS.leave_out_idx != -1:
	deco_print('Leave One Covariate out: %s' %dl._idx2covariate[FLAGS.leave_out_idx])
deco_print('Data Layer Created')
###

### Create Model
deco_print('Creating Model')
if FLAGS.mode == 'train':
	config = Config(feature_dim=291, num_category=7, dropout=0.9)
	model = Model(config)
	config_valid = Config(feature_dim=291, num_category=7, dropout=1.0)
	model_valid = Model(config_valid, force_var_reuse=True, is_training=False)
elif FLAGS.mode == 'test':
	config = Config(feature_dim=291, num_category=7, dropout=1.0)
	model = Model(config, is_training=False)
elif FLAGS.mode == 'sens_anlys' or FLAGS.mode == 'sens_anlys_pair' or FLAGS.mode == 'sens_anlys_trio':
	config = Config(feature_dim=291, num_category=7, dropout=1.0)
	model = Model(config, is_training=False, is_analysis=True)
deco_print('Read Following Config')
deco_print_dict(vars(config))
deco_print('Model Created')
###

with tf.Session() as sess:
	saver = tf.train.Saver(max_to_keep=50)
	if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
		saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
		deco_print('Restored Checkpoint')
	else:
		sess.run(tf.global_variables_initializer())
		deco_print('Random Initialization')

	if FLAGS.mode == 'train':
		deco_print('Executing Training Mode\n')
		tf.summary.scalar(name='loss', tensor=model._loss)
		tf.summary.scalar(name='learning_rate', tensor=model._lr)
		summary_op = tf.summary.merge_all()
		sw = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

		cur_epoch_step = 0
		total_epoch_step_loss = 0.0
		count_epoch_step = 0
		
		for epoch in range(FLAGS.num_epochs):
			epoch_start = time.time()
			total_train_loss = 0.0
			count = 0
			for i, (x, y, info) in enumerate(dl.iterate_one_epoch(model._config.batch_size)):
				print(dl._leave_out_idx)
				print(x[:,FLAGS.leave_out_idx])
				print(dl._idx2covariate[FLAGS.leave_out_idx])
				print(x[:,dl._leave_out_idx])
				dgds
				feed_dict = {model._x_placeholder:x, model._y_placeholder:y, model._epoch_step:info['epoch_step']}
				loss_i, _ = sess.run(fetches=[model._loss, model._train_op], feed_dict=feed_dict)
				total_train_loss += loss_i
				total_epoch_step_loss += loss_i
				count += 1
				count_epoch_step += 1

				if info['epoch_step'] != cur_epoch_step:
					sm, = sess.run(fetches=[summary_op], feed_dict=feed_dict)
					sw.add_summary(sm, global_step=cur_epoch_step)
					train_epoch_step_loss = total_epoch_step_loss / count_epoch_step
					train_loss_value_epoch_step = summary_pb2.Summary.Value(tag='epoch_step_loss', simple_value=train_epoch_step_loss)
					summary = summary_pb2.Summary(value=[train_loss_value_epoch_step])
					sw.add_summary(summary, global_step=cur_epoch_step)
					sw.flush()
					epoch_last = time.time() - epoch_start
					time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
					deco_print('Epoch Step Loss: %f, Elapse / Estimate: %.2fs / %.2fs     ' %(train_epoch_step_loss, epoch_last, time_est), end='\r')
					total_epoch_step_loss = 0.0
					count_epoch_step = 0
					cur_epoch_step = info['epoch_step']

			train_loss = total_train_loss / count
			deco_print('Epoch {} Training Loss: {}                              '.format(epoch, train_loss))
			train_loss_value = summary_pb2.Summary.Value(tag='Train_Epoch_Loss', simple_value=train_loss)
			summary = summary_pb2.Summary(value=[train_loss_value])
			sw.add_summary(summary=summary, global_step=epoch)
			sw.flush()
			epoch_end = time.time()
			deco_print('Did Epoch {} In {} Seconds '.format(epoch, epoch_end - epoch_start))
			
			deco_print('Running Validation')
			total_valid_loss = 0.0
			count_valid = 0
			for i, (x, y, _) in enumerate(dl_valid.iterate_one_epoch(model_valid._config.batch_size)):
				feed_dict = {model_valid._x_placeholder:x, model_valid._y_placeholder:y}
				loss_i, = sess.run(fetches=[model_valid._loss], feed_dict=feed_dict)
				total_valid_loss += loss_i
				count_valid += 1
			valid_loss = total_valid_loss / count_valid
			deco_print('Epoch {} Validation Loss: {}'.format(epoch, valid_loss))
			valid_loss_value = summary_pb2.Summary.Value(tag='Train_Epoch_Valid_Loss', simple_value=valid_loss)
			summary = summary_pb2.Summary(value=[valid_loss_value])
			sw.add_summary(summary=summary, global_step=epoch)
			sw.flush()
			deco_print('Saving Epoch Checkpoint\n')
			saver.save(sess, save_path=os.path.join(FLAGS.logdir, 'model-epoch'), global_step=epoch)

	elif FLAGS.mode == 'test':
		deco_print('Executing Test Mode\n')
		epoch_start = time.time()
		cur_epoch_step = 0
		total_test_loss = 0.0
		count = 0
		for i, (x, y, info) in enumerate(dl.iterate_one_epoch(model._config.batch_size)):
			feed_dict = {model._x_placeholder:x, model._y_placeholder:y}
			loss_i, = sess.run(fetches=[model._loss], feed_dict=feed_dict)
			total_test_loss += loss_i
			count += 1

			if info['epoch_step'] != cur_epoch_step:
				epoch_last = time.time() - epoch_start
				time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
				deco_print('Test Loss: %f, Elapse / Estimate: %.2fs / %.2fs     ' %(total_test_loss / count, epoch_last, time_est), end='\r')
				cur_epoch_step = info['epoch_step']

		test_loss = total_test_loss / count
		deco_print('Test Loss: %f' %test_loss)
		with open(os.path.join(FLAGS.logdir, 'loss.txt'), 'w') as f:
			f.write('Test Loss: %f\n' %test_loss)

	elif FLAGS.mode == 'sens_anlys':
		deco_print('Executing Sensitivity Analysis Mode\n')

		if not os.path.exists(os.path.join(FLAGS.logdir, 'ave_absolute_gradient.npy')):
			count = np.zeros(shape=(5,), dtype=int)
			gradients = np.zeros(shape=(5, model._config.num_category, model._config.feature_dim), dtype=float)
			epoch_start = time.time()
			cur_epoch_step = 0
			sample_step = 0
			for _, (x, y, info, x_cur) in enumerate(dl.iterate_one_epoch(model._config.batch_size, output_current_status=True)):
				if sample_step != FLAGS.sample_size:
					count += np.sum(x_cur, axis=0)
					feed_dict = {model._x_placeholder:x, model._y_placeholder:y}
					gradients_i, = sess.run(fetches=[model._x_gradients], feed_dict=feed_dict)
					for v in range(model._config.num_category):
						gradients_i_v = gradients_i[v]
						gradients[:,v,:] += x_cur.T.dot(np.absolute(gradients_i_v))
					sample_step += 1
				if info['epoch_step'] != cur_epoch_step:
					epoch_last = time.time() - epoch_start
					time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
					deco_print('Elapse / Estimate: %.2fs / %.2fs     ' %(epoch_last, time_est), end='\r')
					cur_epoch_step = info['epoch_step']
					sample_step = 0
			gradients /= count[:, np.newaxis, np.newaxis]
			deco_print('Saving Output in %s' %os.path.join(FLAGS.logdir, 'ave_absolute_gradient.npy'))
			np.save(os.path.join(FLAGS.logdir, 'ave_absolute_gradient.npy'), gradients)

		deco_print('Top 30:')
		top_covariate = feature_ranking(FLAGS.logdir, dl._idx2covariate, float_feature_only=True)
		for item in top_covariate:
			print(item)
		deco_print('Sensitivity Analysis Finished')

	elif FLAGS.mode == 'sens_anlys_pair':
		deco_print('Executing Sensitivity Analysis (Pairs) Mode')
		deco_print('Consider Real-Valued Pairs Only\n')

		### real-valued pairs
		num_feature_pairs = dl._covariate_count_float * (dl._covariate_count_float - 1) // 2
		idx2pair = [(i+dl._covariate_count_int,j+dl._covariate_count_int) for i in range(dl._covariate_count_float-1) for j in range(i+1,dl._covariate_count_float)]
		pair2idx = {idx2pair[l]:l for l in range(num_feature_pairs)}
		###

		if not os.path.exists(os.path.join(FLAGS.logdir, 'ave_absolute_gradient_2.npy')):
			count = np.zeros(shape=(5,), dtype=int)
			gradients = np.zeros(shape=(5, model._config.num_category, num_feature_pairs))
			epoch_start = time.time()
			cur_epoch_step = 0
			sample_step = 0
			for _, (x, y, info, x_cur) in enumerate(dl.iterate_one_epoch(model._config.batch_size, output_current_status=True)):
				if sample_step != FLAGS.sample_size:
					count += np.sum(x_cur, axis=0)
					f, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x})
					for i, j in idx2pair:
						x_copy_1 = copy.deepcopy(x)
						x_copy_2 = copy.deepcopy(x)
						### 1 delta
						x_copy_1[:,i] *= FLAGS.delta
						f_i, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_1})
						x_copy_2[:,j] *= FLAGS.delta
						f_j, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_2})
						### 2 delta
						x_copy_1[:,j] *= FLAGS.delta
						f_ij, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_1})

						finite_diff = np.absolute(f_ij - f_i - f_j + f)
						gradients[:,:,pair2idx[(i,j)]] += x_cur.T.dot(finite_diff)
					sample_step += 1
				if info['epoch_step'] != cur_epoch_step:
					epoch_last = time.time() - epoch_start
					time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
					deco_print('Elapse / Estimate: %.2fs / %.2fs     ' %(epoch_last, time_est), end='\r')
					cur_epoch_step = info['epoch_step']
					sample_step = 0
			gradients /= count[:, np.newaxis, np.newaxis]
			deco_print('Saving Output in %s' %os.path.join(FLAGS.logdir, 'ave_absolute_gradient_2.npy'))
			np.save(os.path.join(FLAGS.logdir, 'ave_absolute_gradient_2.npy'), gradients)

		deco_print('Top 30: ')
		top_covariate = feature_ranking_pair(FLAGS.logdir, dl._idx2covariate, idx2pair)
		for item in top_covariate:
			print(item)
		deco_print('Sensitivity Analysis (Pairs) Finished')

	elif FLAGS.mode == 'sens_anlys_trio':
		deco_print('Executing Sensitivity Analysis (Trios) Mode')
		deco_print('Consider Real-Valued Trios With Important Variables Only\n')

		### real-valued trios
		num_import_features = 20
		important_features = feature_ranking(FLAGS.logdir, dl._idx2covariate, num=num_import_features, float_feature_only=True)
		num_feature_trios = num_import_features * (num_import_features - 1) * (num_import_features - 2) // 6
		idx2trio = [(important_features[i][0],important_features[j][0],important_features[k][0]) for i in range(num_import_features - 2) for j in range(i+1,num_import_features-1) for k in range(j+1,num_import_features)]
		trio2idx = {idx2trio[l]:l for l in range(num_feature_trios)}
		###

		if not os.path.exists(os.path.join(FLAGS.logdir, 'ave_absolute_gradient_3.npy')):
			count = np.zeros(shape=(5,), dtype=int)
			gradients = np.zeros(shape=(5, model._config.num_category, num_feature_trios))
			epoch_start = time.time()
			cur_epoch_step = 0
			sample_step = 0
			for _, (x, y, info, x_cur) in enumerate(dl.iterate_one_epoch(model._config.batch_size, output_current_status=True)):
				if sample_step != FLAGS.sample_size:
					count += np.sum(x_cur, axis=0)
					f, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x})
					for i, j, k in idx2trio:
						x_copy_1 = copy.deepcopy(x)
						x_copy_2 = copy.deepcopy(x)
						x_copy_3 = copy.deepcopy(x)
						### 1 delta
						x_copy_1[:,i] *= FLAGS.delta
						f_i, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_1})
						x_copy_2[:,j] *= FLAGS.delta
						f_j, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_2})
						x_copy_3[:,k] *= FLAGS.delta
						f_k, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_3})
						### 2 delta
						x_copy_1[:,j] *= FLAGS.delta
						f_ij, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_1})
						x_copy_2[:,k] *= FLAGS.delta
						f_jk, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_2})
						x_copy_3[:,i] *= FLAGS.delta
						f_ki, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_3})
						### 3 delta
						x_copy_1[:,k] *= FLAGS.delta
						f_ijk, = sess.run(fetches=[model._prob], feed_dict={model._x_placeholder:x_copy_1})

						finite_diff = np.absolute(f_ijk - f_ij - f_jk - f_ki + f_i + f_j + f_k - f)
						gradients[:,:,trio2idx[(i,j,k)]] += x_cur.T.dot(finite_diff)
					sample_step += 1
				if info['epoch_step'] != cur_epoch_step:
					epoch_last = time.time() - epoch_start
					time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
					deco_print('Elapse / Estimate: %.2fs / %.2fs     ' %(epoch_last, time_est), end='\r')
					cur_epoch_step = info['epoch_step']
					sample_step = 0
			gradients /= count[:, np.newaxis, np.newaxis]
			deco_print('Saving Output in %s' %os.path.join(FLAGS.logdir, 'ave_absolute_gradient_3.npy'))
			np.save(os.path.join(FLAGS.logdir, 'ave_absolute_gradient_3.npy'), gradients)

		deco_print('Top 30: ')
		top_covariate = feature_ranking_trio(FLAGS.logdir, dl._idx2covariate, idx2trio)
		for item in top_covariate:
			print(item)
		deco_print('Sensitivity Analysis (Trios) Finished')

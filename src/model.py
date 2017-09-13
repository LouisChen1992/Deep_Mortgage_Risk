import time, os
from utils import *
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.core.framework import summary_pb2

class Config:
	def __init__(self,
				feature_dim,
				num_category,
				hidden_dim=[200, 140, 140, 140, 140],
				learning_rate=0.1,
				momentum=0.9,
				decay_rate=800,
				batch_size=4000,
				regularization=0.0,
				dropout=1.0):
		self._feature_dim = feature_dim
		self._num_category = num_category
		self._hidden_dim = hidden_dim
		self._learning_rate = learning_rate
		self._momentum = momentum
		self._decay_rate = decay_rate
		self._batch_size = batch_size
		self._num_layer = len(hidden_dim)
		self._regularization = regularization
		self._dropout = dropout

	@property
	def feature_dim(self):
		return self._feature_dim

	@property
	def num_category(self):
		return self._num_category

	@property
	def hidden_dim(self):
		return self._hidden_dim

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def momentum(self):
		return self._momentum

	@property
	def decay_rate(self):
		return self._decay_rate

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def num_layer(self):
		return self._num_layer

	@property
	def regularization(self):
		return self._regularization

	@property
	def dropout(self):
		return self._dropout

class Model:
	def __init__(self, config):
		self._config = config
		self._build_forward_pass_graph()
		self._add_loss()
		self._add_train_op()

	def _build_forward_pass_graph(self):
		self._x_placeholder = tf.placeholder(dtype=tf.float32, shape=(self._config.batch_size, self._config.feature_dim), name='input_placeholder')
		self._y_placeholder = tf.placeholder(dtype=tf.int32, shape=(self._config.batch_size,), name='output_placeholder')

		h_l = self._x_placeholder
		for l in range(self._config.num_layer):
			with tf.variable_scope('dense_layer%d' %l):
				layer_l = Dense(units=self._config.hidden_dim[l], activation=tf.nn.relu)
				h_l = layer_l(h_l)
				h_l = tf.nn.dropout(h_l, self._config.dropout)

		with tf.variable_scope('last_dense_layer'):
			layer = Dense(units=self._config.num_category)
			self._logits = layer(h_l)

		self._predict = tf.argmax(self._logits, axis=-1)

	def _add_loss(self):
		self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self._y_placeholder))

	def _add_train_op(self):
		loss = self._loss

		deco_print('Trainable Variables')
		for var in tf.trainable_variables():
			loss += self._config.regularization * tf.nn.l2_loss(var)
			deco_print('Name: {} and shape: {}'.format(var.name, var.get_shape()))

		self._epoch_step = tf.placeholder(dtype=tf.float32, shape=(), name='epoch_step')
		self._lr = self._config.learning_rate / (1 + self._epoch_step / self._config.decay_rate)

		optimizer = tf.train.MomentumOptimizer(self._lr, self._config.momentum)
		self._train_op = optimizer.minimize(loss)

	def train(self, sess, data_layer, num_epochs, logdir, saver, saver_step):
		deco_print('Executing Training Mode\n')
		tf.summary.scalar(name='loss', tensor=self._loss)
		tf.summary.scalar(name='learning_rate', tensor=self._lr)
		summary_op = tf.summary.merge_all()
		sw = tf.summary.FileWriter(logdir, sess.graph)

		cur_epoch_step = 0
		total_epoch_step_loss = 0.0
		count_epoch_step = 0
		
		for epoch in range(num_epochs):
			epoch_start = time.time()
			total_train_loss = 0.0
			count = 0
			for i, (x, y, info) in enumerate(data_layer.iterate_one_epoch(self._config.batch_size)):
				feed_dict = {self._x_placeholder:x, self._y_placeholder:y, self._epoch_step:info['epoch_step']}
				loss_i, _ = sess.run(fetches=[self._loss, self._train_op], feed_dict=feed_dict)
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
					if cur_epoch_step % 100 == 0:
						saver_step.save(sess, save_path=os.path.join(logdir, 'model-epoch-step'), global_step=cur_epoch_step)

			train_loss = total_train_loss / count
			deco_print('Epoch {} Training Loss: {}                              '.format(epoch, train_loss))
			train_loss_value = summary_pb2.Summary.Value(tag='Train_Epoch_Loss', simple_value=train_loss)
			summary = summary_pb2.Summary(value=[train_loss_value])
			sw.add_summary(summary=summary, global_step=epoch)
			sw.flush()
			epoch_end = time.time()
			deco_print('Did Epoch {} In {} Seconds '.format(epoch, epoch_end - epoch_start))
			deco_print('Saving Epoch Checkpoint\n')
			saver.save(sess, save_path=os.path.join(logdir, 'model-epoch'), global_step=epoch)

	def test(self, sess, data_layer):
		deco_print('Executing Test Mode\n')
		epoch_start = time.time()
		cur_epoch_step = 0
		total_test_loss = 0.0
		count = 0
		for i, (x, y, info) in enumerate(data_layer.iterate_one_epoch(self._config.batch_size)):
			feed_dict = {self._x_placeholder:x, self._y_placeholder:y}
			loss_i, = sess.run(fetches=[self._loss], feed_dict=feed_dict)
			total_test_loss += loss_i
			count += 1

			if info['epoch_step'] != cur_epoch_step:
				epoch_last = time.time() - epoch_start
				time_est = epoch_last / (info['idx_file'] + 1) * info['num_file']
				deco_print('Test Loss: %f, Elapse / Estimate: %.2fs / %.2fs     ' %(total_test_loss / count, epoch_last, time_est), end='\r')
				cur_epoch_step = info['epoch_step']

		test_loss = total_test_loss / count
		deco_print('Test Loss: %f' %test_loss)

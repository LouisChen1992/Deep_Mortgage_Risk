import tensorflow as tf
from model import *

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tr.flags.DEFINE_string('mode', 'train', 'Mode')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of training epochs')
FLAGS = tf.flags.FLAGS

deco_print('Creating Data Layer')
###
pass
deco_print('Data Layer Created')
###

deco_print('Creating Model')
config = Config()
model = Model(config)
deco_print('Model Created')

if FLAGS.model == 'train':
	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=100)
		if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
			saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
			deco_print('Restored checkpoint. Resuming training')
		else:
			sess.run(tf.global_variables_initializer())
			deco_print('Random initialization')
		model.train(sess, data_layer, FLAGS.num_epochs, FLAGS.logdir, saver)

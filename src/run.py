import tensorflow as tf
from model import *
from data_layer import *
from utils import *

tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('mode', 'train', 'Mode')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of training epochs')
FLAGS = tf.flags.FLAGS

if FLAGS.mode == 'train':
	deco_print('Creating Data Layer')
	path = '/vol/Numpy_data_subprime_new'
	dl = DataInRamInputLayer(path=path, mode='train')
	deco_print('Data Layer Created')
	

	deco_print('Creating Model')
	config = Config(feature_dim=291, num_category=7, dropout=0.9)
	deco_print('Read Following Config')
	deco_print_dict(vars(config))
	model = Model(config)
	deco_print('Model Created')

	with tf.Session() as sess:
		saver = tf.train.Saver(max_to_keep=50)
		saver_step = tf.train.Saver(max_to_keep=100)
		if tf.train.latest_checkpoint(FLAGS.logdir) is not None:
			saver.restore(sess, tf.train.latest_checkpoint(FLAGS.logdir))
			deco_print('Restored checkpoint. Resuming training')
		else:
			sess.run(tf.global_variables_initializer())
			deco_print('Random initialization')
		model.train(sess, dl, FLAGS.num_epochs, FLAGS.logdir, saver, saver_step)

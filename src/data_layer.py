import os
import numpy as np

class DataInRamInputLayer():
	def __init__(self, path, mode):
		self._path = path
		self._mode = mode
		self._create_file_list()

	def _create_file_list(self):
		if self._mode == 'train':
			X_int_list = []
			X_float_list = []
			outcome_list = []
			for file in os.listdir(self._path):
				if file.startswith('X_data_np_int'):
					X_int_list.append(file)
				elif file.startswith('X_data_np_float'):
					X_float_list.append(file)
				elif file.startswith('outcome'):
					outcome_list.append(file)
			self._X_int_list = sorted(X_int_list)
			self._X_float_list = sorted(X_float_list)
			self._outcome_list = sorted(outcome_list)
			self._num_file = len(self._X_int_list)

			###
			self._epoch_step = 0
			###

	def iterate_one_epoch(self, batch_size):
		outseq = np.arange(self._num_file)
		np.random.shuffle(outseq)
		for idx in range(len(outseq)):
			idx_file = outseq[i]
			X_int = np.load(os.path.join(self._path, self._X_int_list[idx_file]))
			X_float = np.load(os.path.join(self._path, self._X_float_list[idx_file]))
			outcome = np.load(os.path.join(self._path, self._outcome_list[idx_file]))

			num_example = X_int.shape[0]
			num_batch = num_example // batch_size
			idx_example = np.arange(num_example)
			np.random.shuffle(idx_example)
			for idx_batch in range(num_batch):
				X_int_input = X_int[idx_batch*batch_size:(idx_batch+1)*batch_size]
				X_float_input = X_float[idx_batch*batch_size:(idx_batch+1)*batch_size]
				X_input = np.concatenate((X_int_input, X_float_input), axis=1)
				Y_input = outcome[idx_batch*batch_size:(idx_batch+1)*batch_size]
				if idx_batch == num_batch - 1:
					self._epoch_step += 1
				batch_info = {'epoch_step':self._epoch_step,
					'num_file':self._num_file, 'idx_file':idx}
				yield X_input, Y_input, batch_info
import os
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class DataInRamInputLayer():
	def __init__(self, path, shuffle=False, load_file_list=True, leave_out_idx=-1):
		self._path = path
		self._shuffle = shuffle
		self._create_covariate_idx_associations()
		if load_file_list:
			self._create_file_list()
			self._epoch_step = 0
		self._leave_out_idx = leave_out_idx

	def _create_covariate_idx_associations(self):
		with open('src/covariate2idx_int.json', 'r') as f:
			self._covariate2idx_int = json.load(f)
			self._idx2covariate_int = {value:key for key, value in self._covariate2idx_int.items()}
			self._covariate_count_int = len(self._covariate2idx_int.keys())

		with open('src/covariate2idx_float.json', 'r') as f:
			self._covariate2idx_float = json.load(f)
			self._idx2covariate_float = {value:key for key, value in self._covariate2idx_float.items()}
			self._covariate_count_float = len(self._covariate2idx_float.keys())

		self._idx2covariate = {}
		for key in self._idx2covariate_int.keys():
			self._idx2covariate[key] = self._idx2covariate_int[key]
		for key in self._idx2covariate_float.keys():
			self._idx2covariate[key+self._covariate_count_int] = self._idx2covariate_float[key]
		self._covariate_count = self._covariate_count_int + self._covariate_count_float
		self._covariate2idx = {value:key for key, value in self._idx2covariate.items()}

		with open('src/outcome2idx.json', 'r') as f:
			self._outcome2idx = json.load(f)
			self._idx2outcome = {value:key for key, value in self._outcome2idx.items()}
			self._outcome_count = len(self._outcome2idx.keys())

	def _create_file_list(self):
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
		self._outseq = np.arange(self._num_file)

	def _construct_polynomial_feature(self, feature_in, poly_order=1, include_bias=False):
		poly = PolynomialFeatures(degree=poly_order, include_bias=include_bias)
		feature_out = poly.fit_transform(feature_in)
		return feature_out

	def iterate_one_epoch(self, batch_size, output_current_status=False, poly_order=1):
		if self._shuffle:
			np.random.shuffle(self._outseq)

		for idx in range(self._num_file):
			idx_file = self._outseq[idx]
			X_int = np.load(os.path.join(self._path, self._X_int_list[idx_file]))
			X_float = np.load(os.path.join(self._path, self._X_float_list[idx_file]))
			outcome = np.load(os.path.join(self._path, self._outcome_list[idx_file]))

			num_example = X_int.shape[0]
			num_batch = num_example // batch_size
			idx_example = np.arange(num_example)
			if self._shuffle:
				np.random.shuffle(idx_example)

			for idx_batch in range(num_batch):
				idx_input = idx_example[idx_batch*batch_size:(idx_batch+1)*batch_size]
				X_int_input = X_int[idx_input]
				X_float_input = X_float[idx_input]
				X_input = np.concatenate((X_int_input, X_float_input), axis=1)
				### construct polynomial feature
				# X_input = self._construct_polynomial_feature(X_input, poly_order=poly_order)
				###
				Y_input = outcome[idx_input]

				if idx_batch == num_batch - 1:
					self._epoch_step += 1
				batch_info = {'epoch_step':self._epoch_step,
					'num_file':self._num_file, 'idx_file':idx}

				if self._leave_out_idx != -1:
					X_input[:, self._leave_out_idx] = 0.0
					print(X_input[:,251])

				if not output_current_status:
					yield X_input, Y_input, batch_info
				else:
					### output current status
					X_current_status = X_int_input[:,:5]
					yield X_input, Y_input, batch_info, X_current_status

	def calculate_feature_statistics(self):
		moments = np.zeros((2, self._covariate_count))
		count = 0
		self._max = np.array([-float('inf')] * self._covariate_count)
		self._min = np.array([float('inf')] * self._covariate_count)
		for idx_file in range(self._num_file):
			X_int = np.load(os.path.join(self._path, self._X_int_list[idx_file]))
			X_float = np.load(os.path.join(self._path, self._X_float_list[idx_file]))
			assert(X_int.shape[0] == X_float.shape[0])
			count += X_int.shape[0]

			X = np.concatenate([X_int, X_float], axis=1)
			moments[0] += np.sum(X, axis=0)
			moments[1] += np.sum(X**2, axis=0)
			self._max = np.maximum(np.max(X, 0), self._max)
			self._min = np.minimum(np.min(X, 0), self._min)
		self._mean = moments[0] / count
		self._std = np.sqrt(moments[1] / count - self._mean ** 2)

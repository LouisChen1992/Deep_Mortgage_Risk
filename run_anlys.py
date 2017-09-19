import numpy as np
from src.utils_anlys import covariate_ranking_by_ave_absolute_gradient
from src.data_layer import DataInRamInputLayer

path = '/vol/Numpy_data_subprime_Test_new'
mode = 'anlys'
dl = DataInRamInputLayer(path=path, mode=mode)
ave_absolute_gradient = np.load('model/ave_absolute_gradient.npy')
ave_absolute_gradient_sort = covariate_ranking_by_ave_absolute_gradient(dl._idx2covariate, ave_absolute_gradient)
for item in ave_absolute_gradient_sort:
	print(item)
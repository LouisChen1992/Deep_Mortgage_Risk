import numpy as np

def covariate_ranking_by_ave_absolute_gradient(idx2covariate, ave_absolute_gradient, state=(0,1)):
	gradient = ave_absolute_gradient[state[0]][state[1]]
	gradient_sort = sorted([(i,gradient[i]) for i in range(len(gradient))], key=lambda t:-t[1])
	return [(idx2covariate{i}, grad) for (i, grad) in gradient_sort if i >= 5]
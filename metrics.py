import numpy as np


# https://gist.github.com/bwhite/3726239


def precision_at_k(r, k):
	assert k >= 1
	r = np.asarray(r)[:k] != 0
	if r.size != k:
		raise ValueError('Relevance score length < k')
	return np.mean(r)


def average_precision(r):
	r = np.asarray(r) != 0
	out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
	if not out:
		return 0.
	return np.mean(out)


def mean_average_precision(rs):
	return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.


def ndcg_at_k(r, k, method=0):
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	if not dcg_max:
		return 0.
	return dcg_at_k(r, k, method) / dcg_max

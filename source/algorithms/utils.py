import numpy as np
from scipy.stats import cauchy


def stochastic_universal_selection(P, ps):
	space = 1.0 / ps
	sumP = 0
	rnd = np.random.rand()
	rndn = [rnd + i * space for i in range(ps)]
	label = [1] * ps
	index = [0] * ps
	for j in range(len(P)):
		sumP += P[j]
		for i in range(ps):
			nlabel = int(not label[i])
			label[i] = nlabel & (rndn[i] < sumP)
			index[i] += label[i] * j
			label[i] |= index[i]
	np.random.shuffle(index)
	return index


def generate_cauchy(loc, scale, m=1):
	loc = np.full(m, loc)
	scale = np.full(m, scale)
	rv_cauchy = cauchy(loc=loc, scale=scale)
	cauchy_samples = rv_cauchy.rvs(size=m)
	while np.any(cauchy_samples < 0):
		indices = np.where(cauchy_samples < 0)[0]
		rv_cauchy = cauchy(loc=loc[indices], scale=scale[indices])
		cauchy_samples[indices] = rv_cauchy.rvs(size=len(indices))
	cauchy_samples_clipped = np.clip(cauchy_samples, 0, 1)
	return cauchy_samples_clipped


def random_indexes(n, size, ignore=[]):
	indexes = [pos for pos in range(size) if pos not in ignore]

	assert len(indexes) >= n
	np.random.shuffle(indexes)

	if n == 1:
		return indexes[0]
	else:
		return indexes[:n]


def cross_clip(bounds, solution, original):
	min_bound, max_bound = np.array(bounds).T
	clip_sol = np.clip(solution, min_bound, max_bound)

	if np.all(solution == clip_sol):
		return solution

	idx_lowest = (solution < min_bound)
	solution[idx_lowest] = (original[idx_lowest] + min_bound[idx_lowest]) / 2.0
	idx_upper = (solution > max_bound)
	solution[idx_upper] = (original[idx_upper] + max_bound[idx_upper]) / 2.0
	return solution


def update_CR(SCR, weights):
	total = np.sum(weights)
	weights = weights / total
	CRnew = np.sum(weights * SCR * SCR) / np.sum(weights * SCR)
	CRnew = np.clip(CRnew, 0, 1)
	return CRnew


def update_F(SF, weights):
	total = np.sum(weights)
	weights = weights / total
	Fnew = np.sum(weights * SF * SF) / np.sum(weights * SF)
	Fnew = np.clip(Fnew, 0, 1)
	return Fnew

def limit_memory(memory, memorySize):
	"""
	Limit the memory to  the memorySize
	"""
	memory = np.array(memory)

	if len(memory) > memorySize:
		indexes = np.random.permutation(len(memory))[:memorySize]
		memory = memory[indexes]

	return memory.tolist()

def limit_population(population, memorySize, best_index, f):
	"""
	Limit the memory to  the memorySize
	"""

	if len(population) > memorySize:
		indexes = np.random.permutation(len(population))[:memorySize]
		while best_index not in indexes:
			indexes = np.random.permutation(len(population))[:memorySize]
		population = population[indexes]
		f = f[indexes]
	return population, f



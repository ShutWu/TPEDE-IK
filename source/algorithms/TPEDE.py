import numpy as np
from algorithms.utils import stochastic_universal_selection, generate_cauchy, random_indexes, cross_clip, update_CR, update_F, limit_memory, limit_population


class TPEDE_solver:
	def __init__(self, obj_func, bounds, max_FES=50000):
		self.obj_func = obj_func
		self.bounds = bounds
		self.max_FES = max_FES
		self.D = len(bounds)
		self.NP_ini = int(25 * np.log(self.D) * np.sqrt(self.D))
		self.current_fes = self.NP_ini
		self.KCR = 4
		self.NP = self.NP_ini
		self.uF = 0.5
		self.uCR = [0.9] * self.KCR
		self.r = np.zeros(self.KCR)
		self.ct = np.zeros(self.NP, dtype=int)
		self.PCR = np.full(self.KCR, 1/self.KCR)
		self.flag = False
		self.pmin = 0.1
		self.r = np.zeros(self.KCR)
		self.G = 0
		# Initialization
		self.min_bounds = np.array([bound[0] for bound in bounds])
		self.max_bounds = np.array([bound[1] for bound in bounds])
		self.x = self.min_bounds + (self.max_bounds - self.min_bounds) * np.random.rand(self.NP_ini, self.D)
		self.fitness = np.apply_along_axis(self.obj_func, axis=1, arr=self.x)
		self.memB = self.x.tolist().copy()

	def terminate(self):
		if self.current_fes >= self.max_FES:
			return True
		return False

	def step(self):
		memory_size_B = self.NP * 3
		SF = []
		SCR = []
		u = np.zeros((self.NP, self.D))
		ns = np.zeros(self.NP)
		self.uCR[-1] = 0.2
		index = np.array(stochastic_universal_selection(self.PCR, self.NP))

		if not self.flag or self.current_fes > 0.3 * self.max_FES:
			CR = np.zeros(self.NP)
			for cate in range(self.KCR):
				index_i = np.where(index == cate)[0]
				if len(index_i) == 0:
					continue
				rand_CR = np.random.normal(self.uCR[cate], 0.1, size=len(index_i))
				CR[index_i] = rand_CR
			F = generate_cauchy(self.uF, 0.1, self.NP)
		else:
			index1 = np.argsort(self.fitness)
			self.x = self.x[index1, :]
			self.fitness = self.fitness[index1]
			F = generate_cauchy((np.arange(self.NP) + 1) / self.NP + 0.1, 0.5, self.NP)
			CR = np.random.normal((np.arange(self.NP) + 1) / self.NP, 0.1, self.NP)
			self.flag = False

		CR = np.clip(CR, 0, 1)
		if self.current_fes < 0.2 * self.max_FES:
			CR = np.maximum(CR, 0.9)
		elif self.current_fes < 0.67 * self.max_FES:
			CR = np.maximum(CR, 0.4)
		else:
			CR = np.maximum(CR, 0.8)

		if self.current_fes < 0.66 * self.max_FES:
			F = np.clip(F, 0.3, 0.8)
		if self.current_fes < 0.3 * self.max_FES:
			Fw = 1.4 * F
		else:
			Fw = 1.0 * F

		for i in range(self.NP):
			pmax = 0.25 - 0.2 * self.current_fes / self.max_FES
			p = np.random.rand() * (pmax - self.pmin) + self.pmin
			maxbest = int(p * self.NP) + 1
			bests = np.argsort(self.fitness)[:maxbest]
			pbest = np.random.choice(bests)
			xbest = self.x[pbest]

			r1 = random_indexes(1, self.NP - 1, ignore=[pbest, i])
			xr1 = self.x[r1]
			if self.current_fes < 0.33 * self.max_FES:
				r2 = random_indexes(1, len(self.memB)+self.NP, ignore=[pbest, i, r1])
				if r2 < len(self.memB):
					xr2 = self.memB[r2]
				else:
					xr2 = self.x[r2 - len(self.memB)]
			else:
				r2 = random_indexes(1, self.NP, ignore=[pbest, i, r1])
				xr2 = self.x[r2]

			v = self.x[i] + F[i] * (xbest - self.x[i]) + Fw[i] * (xr1 - xr2)

			cross_points = np.random.rand(self.D) <= CR[i]
			random_index = np.random.randint(self.D)
			cross_points[random_index] = True
			u[i] = np.where(cross_points, v, self.x[i])
			u[i] = cross_clip(self.bounds, u[i], self.x[i])

		self.memB.extend(self.x.tolist())

		weights = []
		fitness = np.apply_along_axis(self.obj_func, axis=1, arr=u)
		for i in range(self.NP):
			if fitness[i] <= self.fitness[i]:
				if fitness[i] < self.fitness[i]:
					SF.append(F[i])
					SCR.append(CR[i])
					loc = u[i] - self.x[i]
					delta_loc = np.std(loc[np.nonzero(loc)]) + 1e-8
					weights.append(delta_loc)
					ns[i] = 1
				self.x[i], self.fitness[i] = u[i].copy(), fitness[i].copy()
			self.current_fes += 1

		if len(SCR) > 0:
			for cate in range(self.KCR):
				index_i = np.where(index == cate)[0]
				if len(index_i) == 0:
					continue
				nsk = np.sum(ns[index_i])
				self.r[cate] = (nsk ** 2) / (np.sum(ns) * len(index_i))
			for cate in range(self.KCR):
				self.PCR[cate] = self.r[cate] / np.sum(self.r)

			self.uCR[self.G % self.KCR] = update_CR(SCR, weights)
			self.uF = update_F(SF, weights)

		else:
			self.uCR[self.G % self.KCR] = np.random.normal(0.25, 0.1)
			self.uF = np.clip(np.random.normal(self.uF, 0.1), 0.1, 0.5)
			self.flag = True

		self.memB = limit_memory(self.memB, memory_size_B)

		if not self.flag:
			self.NP = int(self.NP_ini - (self.NP_ini - self.KCR) * self.current_fes / self.max_FES)

		best_index = np.argmin(self.fitness)
		self.x, self.fitness = limit_population(self.x, self.NP, best_index, self.fitness)
		self.G += 1

	def run(self):
		log_fitness = np.zeros(self.max_FES)
		log_fitness[:self.NP] = np.min(self.fitness)
		point = self.NP
		while not self.terminate():
			self.step()

			current_best = np.min(self.fitness)
			if current_best < log_fitness[point - 1]:
				log_fitness[point:self.current_fes] = current_best
			else:
				log_fitness[point:self.current_fes] = log_fitness[point - 1]
			point = self.current_fes
		return log_fitness



import numpy as np
import quaternion

class objective_function_Panda_quaternion:
	def __init__(self, target_theta):
		self.t_0_1 = np.quaternion(0, 0, 0, 0) * 1e-3
		self.t_1_2 = np.quaternion(0, -333, 0, 0) * 1e-3
		self.t_2_3 = np.quaternion(0, 0, 0, 0) * 1e-3
		self.t_3_4 = np.quaternion(0, -316, 0, 82) * 1e-3
		self.t_4_5 = np.quaternion(0, -82, 0, 384) * 1e-3
		self.t_5_6 = np.quaternion(0, 0, 0, 0) * 1e-3
		self.t_6_7 = np.quaternion(0, -88, 0, 107) * 1e-3
		self.unu = np.quaternion(1, 0, 0, 0)
		self.zero = np.quaternion(0, 0, 0, 0)

		self.target_theta = target_theta
		self.R_target, self.T_target = self.forward(target_theta)

	@staticmethod
	def quatrans(R, T, t):
		return R * T * R.conj() + t

	def forward(self, theta):
		r0_1 = np.quaternion(np.cos(theta[0] / 2), -np.sin(theta[0] / 2), 0, 0)
		r1_2 = np.quaternion(np.cos(theta[1] / 2), 0, np.sin(theta[1] / 2), 0)
		r2_3 = np.quaternion(np.cos(theta[2] / 2), -np.sin(theta[2] / 2), 0, 0)
		r3_4 = np.quaternion(np.cos(theta[3] / 2), 0, -np.sin(theta[3] / 2), 0)
		r4_5 = np.quaternion(np.cos(theta[4] / 2), 0, 0, np.sin(theta[4] / 2))
		r5_6 = np.quaternion(np.cos(theta[5] / 2), 0, np.sin(theta[5] / 2), 0)
		r6_7 = np.quaternion(np.cos(theta[6] / 2), 0, 0, np.sin(theta[6] / 2))
		T6_7 = self.quatrans(self.unu, self.zero, self.t_6_7)
		T5_7 = self.quatrans(r5_6, T6_7, self.t_5_6)
		T4_7 = self.quatrans(r4_5, T5_7, self.t_4_5)
		T3_7 = self.quatrans(r3_4, T4_7, self.t_3_4)
		T2_7 = self.quatrans(r2_3, T3_7, self.t_2_3)
		T1_7 = self.quatrans(r1_2, T2_7, self.t_1_2)
		T0_7 = self.quatrans(r0_1, T1_7, self.t_0_1)
		R0_7 = r0_1 * r1_2 * r2_3 * r3_4 * r4_5 * r5_6 * r6_7
		return R0_7, T0_7

	@staticmethod
	def are_quaternions_equal(q1, q2):
		return np.allclose(q1, q2) or np.allclose(q1, -q2)

	def my_getRO(self, theta1):
		r0_1 = np.quaternion(np.cos(theta1[0] / 2), -np.sin(theta1[0] / 2), 0, 0)
		r1_2 = np.quaternion(np.cos(theta1[1] / 2), 0, np.sin(theta1[1] / 2), 0)
		r2_3 = np.quaternion(np.cos(theta1[2] / 2), -np.sin(theta1[2] / 2), 0, 0)
		r3_4 = np.quaternion(np.cos(theta1[3] / 2), 0, -np.sin(theta1[3] / 2), 0)
		R0_4 = r0_1 * r1_2 * r2_3 * r3_4
		RO = R0_4.conj() * self.R_target
		return RO

	def solve_end3theta(self, theta1_3):
		theta5_7 = np.zeros(3)
		RO = self.my_getRO(theta1_3)
		theta5_7[0] = np.arctan2(RO.z, RO.w) - np.arctan2(RO.x, RO.y)
		theta5_7[2] = np.arctan2(RO.z, RO.w) + np.arctan2(RO.x, RO.y)
		theta5_7[1] = 2 * np.arctan2(RO.y * np.sin((theta5_7[0] + theta5_7[2]) / 2),
		                             RO.z * np.cos((theta5_7[0] - theta5_7[2]) / 2))
		theta5_7 = np.mod(theta5_7 + np.pi, 2 * np.pi) - np.pi
		theta = np.concatenate((theta1_3, theta5_7))
		R, _ = self.forward(theta)
		if self.are_quaternions_equal(R, self.R_target):
			return theta
		else:
			theta[-2] = -theta[-2]
			return theta

	def evaluate(self, theta):
		theta_all = self.solve_end3theta(theta)
		R, T = self.forward(theta_all)
		error = T - self.T_target
		return np.sqrt(error.x ** 2 + error.y ** 2 + error.z ** 2)

import numpy as np
np.random.seed(0)
from objective_functions.Panda_obj_RTversion import objective_function_Panda_quaternion
from algorithms.TPEDE_IK import TPEDE_IK_solver
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


global sims
sims = {}
client = RemoteAPIClient()
sim = client.require('sim')

def moveToConfig(robot, handles, maxVel, maxAccel, maxJerk, targetConf):
	sim = sims[robot]
	currentConf = []
	for i in range(len(handles)):
		currentConf.append(sim.getJointPosition(handles[i]))
	params = {}
	params['joints'] = handles
	params['targetPos'] = targetConf
	params['maxVel'] = maxVel
	params['maxAccel'] = maxAccel
	params['maxJerk'] = maxJerk
	sim.moveToConfig(params)  # one could also use sim.moveToConfig_init, sim.moveToConfig_step and sim.moveToConfig_cleanup

class DrawCircle:
	def __init__(self):
		self.points = []
		self.center = [-0.7, 0, 0.3]
		self.normal = [1, 0, 0]
		self.radius = 0.2

	@staticmethod
	def circle_points(center, normal, radius, num_points=100):
		C = np.array(center)
		n = np.array(normal)
		n = n / np.linalg.norm(n)

		if np.all(n == [0, 0, 1]) or np.all(n == [0, 0, -1]):
			a = np.array([0, 1, 0])
		else:
			a = np.array([0, 0, 1])

		u = np.cross(n, a)
		u = u / np.linalg.norm(u)

		v = np.cross(n, u)
		v = v / np.linalg.norm(v)

		theta = np.linspace(0, 2 * np.pi, num_points)
		circle_points = np.zeros((num_points, 3))

		for i in range(num_points):
			circle_points[i] = C + radius * (u * np.cos(theta[i]) + v * np.sin(theta[i]))

		return circle_points

	def trajectory_points(self):
		points = self.circle_points(self.center, self.normal, self.radius)
		return points


def solve_TPEDE_IK():
	check_bounds = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718 + np.pi / 2, -0.0698 + np.pi / 2)]
	res = []
	obj_func = objective_function_Panda_quaternion(points[0], target_orientation)
	solver = TPEDE_IK_solver(obj_func.evaluate, check_bounds)
	fit, best_theta = solver.run()
	res.append(best_theta)
	for i in range(1, len(points)):
		print("Solving for point ", i)
		check_bounds = [(best_theta[i] - 2 * np.pi / 180, best_theta[i] + 2 * np.pi / 180) for i in range(4)]
		obj_func = objective_function_Panda_quaternion(points[i], target_orientation)
		solver = TPEDE_IK_solver(obj_func.evaluate, check_bounds)
		fit, best_theta = solver.run()
		res.append(best_theta)
	return res

def adjust_theta(theta_current, theta_old):
	adjusted_theta = theta_current + 2 * np.pi * round((theta_old - theta_current) / (2 * np.pi))
	return adjusted_theta

if __name__ == "__main__":
	target_orientation = np.quaternion(np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0)
	points = DrawCircle().trajectory_points()
	data = solve_TPEDE_IK()
	data_res = []
	differences = np.array([0, 0, 0, np.pi / 2, 0, -np.pi, 0])
	obj_func = objective_function_Panda_quaternion(points[0], target_orientation)
	for i in range(len(data)):
		position = obj_func.solve_end3theta(data[i])
		position = (position - differences) * np.array([1, 1, 1, 1, 1, -1, 1])
		if i != 0:
			for i in range(4, 7):
				position[i] = adjust_theta(position[i], old_theta[i])
		else:
			# 将最后三个角度限制在[-pi, pi]之间
			position[-3] = np.mod(position[-3] + np.pi, 2 * np.pi) - np.pi
			position[-2] = np.mod(position[-2], 2 * np.pi)
			position[-1] = np.mod(position[-1] + np.pi, 2 * np.pi) - np.pi
		old_theta = position.copy()
		data_res.append(position)
	data_res = np.array(data_res)

	input('Press Enter to start the simulation...')
	sim.startSimulation()
	robot = 'Franka'
	client = RemoteAPIClient()
	sim = client.require('sim')
	sims[robot] = sim
	sim.setStepping(True)
	jointHandles = []
	for i in range(7):
		jointHandles.append(sim.getObject('/' + robot + '/joint', {'index': i}))

	vel = 110 * np.pi / 180
	accel = 40 * np.pi / 180
	jerk = 80 * np.pi / 180
	end = sim.getObject('/Franka/connection')
	maxVel = [vel, vel, vel, vel, vel, vel, vel]
	maxAccel = [accel, accel, accel, accel, accel, accel, accel]
	maxJerk = [jerk, jerk, jerk, jerk, jerk, jerk, jerk]
	color_points = sim.addDrawingObject(sim.drawing_points, 12, 0.0, -1, 10000, [204 / 255, 36 / 255, 124 / 255])

	for i in range(len(data_res)):
		moveToConfig(robot, jointHandles, maxVel, maxAccel, maxJerk, data_res[i].tolist())
		end_Pos = sim.getObjectPosition(end, -1)
		sim.addDrawingObjectItem(color_points, end_Pos)

	input('Press Enter to stop the simulation...')
	sim.setStepping(False)
	sim.removeDrawingObject(color_points)
	sim.stopSimulation()
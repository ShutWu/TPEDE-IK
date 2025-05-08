import numpy as np
np.random.seed(0)
from objective_functions.Panda_obj import objective_function_Panda_quaternion
from objective_functions.UR5_obj import objective_function_UR5_quaternion
from algorithms.TPEDE_IK import TPEDE_IK_solver
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

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

if __name__ == '__main__':
	manipulator = input('Select the manipulator (Panda, UR5): ')
	if manipulator == 'Panda':
		# One_point IK for Panda
		bounds = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718 + np.pi / 2, -0.0698 + np.pi / 2), (-2.8973, 2.8973), (-0.0175 - np.pi, 3.7525 - np.pi), (-2.8973, 2.8973)]
		dimension_decreased_bounds = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718 + np.pi / 2, -0.0698 + np.pi / 2)]
		theta_target = [np.random.uniform(low=bound[0], high=bound[1]) for bound in bounds]
		objfunc = objective_function_Panda_quaternion(theta_target)
		tpede = TPEDE_IK_solver(objfunc.evaluate, dimension_decreased_bounds)
		_, xbest = tpede.run()
		xbest = objfunc.solve_end3theta(xbest)

		# Control the robot in the simulation software Coppeliasim as example
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

		# Keep the revolute rule as Coppeliasim
		differences = np.array([0, 0, 0, np.pi / 2, 0, -np.pi, 0])
		solved_theta = (xbest - differences) * np.array([1, 1, 1, 1, 1, -1, 1])
		target_theta = (np.array(theta_target) - differences) * np.array([1, 1, 1, 1, 1, -1, 1])

		moveToConfig(robot, jointHandles, maxVel, maxAccel, maxJerk, target_theta.tolist())
		time.sleep(2)
		endPos_target = sim.getObjectPosition(end, -1)  # get the position of the target end effector
		endOrientation_target = sim.getObjectOrientation(end, -1)   # get the orientation of the target end effector

		moveToConfig(robot, jointHandles, maxVel, maxAccel, maxJerk, solved_theta.tolist())
		time.sleep(2)
		endPos_solved = sim.getObjectPosition(end, -1)  # get the position of the end effector
		endOrientation_solved = sim.getObjectOrientation(end, -1)   # get the orientation of the end effector

		position_error = np.linalg.norm(np.array(endPos_target) - np.array(endPos_solved))
		orientation_error = np.linalg.norm(np.array(endOrientation_target) - np.array(endOrientation_solved))
		print('Position error:', position_error)
		print('Orientation error:', orientation_error)
		sim.stopSimulation()
	elif manipulator == 'UR5':
		# One_point IK for Panda
		bounds = [(-np.pi, np.pi)] * 6
		dimension_decreased_bounds = [(-np.pi, np.pi)] * 3
		theta_target = [np.random.uniform(low=bound[0], high=bound[1]) for bound in bounds]
		objfunc = objective_function_UR5_quaternion(theta_target)
		tpede = TPEDE_IK_solver(objfunc.evaluate, dimension_decreased_bounds)
		_, xbest = tpede.run()
		xbest = objfunc.solve_end3theta(xbest)

		# Control the robot in the simulation software Coppeliasim as example
		input('Press Enter to start the simulation...')
		sim.startSimulation()
		robot = 'UR5'
		client = RemoteAPIClient()
		sim = client.require('sim')
		sims[robot] = sim
		sim.setStepping(True)
		jointHandles = []
		for i in range(6):
			jointHandles.append(sim.getObject('/' + robot + '/joint', {'index': i}))

		vel = 110 * np.pi / 180
		accel = 40 * np.pi / 180
		jerk = 80 * np.pi / 180
		end = sim.getObject('/UR5/connection')
		maxVel = [vel, vel, vel, vel, vel, vel]
		maxAccel = [accel, accel, accel, accel, accel, accel]
		maxJerk = [jerk, jerk, jerk, jerk, jerk, jerk]

		solved_theta = xbest
		target_theta = theta_target

		moveToConfig(robot, jointHandles, maxVel, maxAccel, maxJerk, target_theta)
		time.sleep(2)
		endPos_target = sim.getObjectPosition(end, -1)  # get the position of the target end effector
		endOrientation_target = sim.getObjectOrientation(end, -1)  # get the orientation of the target end effector

		moveToConfig(robot, jointHandles, maxVel, maxAccel, maxJerk, solved_theta.tolist())
		time.sleep(2)
		endPos_solved = sim.getObjectPosition(end, -1)  # get the position of the end effector
		endOrientation_solved = sim.getObjectOrientation(end, -1)  # get the orientation of the end effector

		position_error = np.linalg.norm(np.array(endPos_target) - np.array(endPos_solved))
		orientation_error = np.linalg.norm(np.array(endOrientation_target) - np.array(endOrientation_solved))
		print('Position error:', position_error)
		print('Orientation error:', orientation_error)
		sim.stopSimulation()

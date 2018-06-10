from pypot.vrep import from_vrep
import time
import numpy as np

def move(motor_, present_position_, goal_position_):
    stride = 5 if goal_position_ > present_position_ else -5
    for step in np.arange(present_position_, goal_position_, stride):
        motor_.goal_position = step
        time.sleep(0.2)
    motor_.goal_position = goal_position_

class Env(object):
	action_bound = [-1, 1]	# action will be angle move between [-1,1]
	state_dim = 4	# theta1 & theta2, distance to goal,get_point
	action_dim = 2
	arm1l = 0.185
	arm2l = 0.515
	arm1_theta = 0
	arm2_theta = -10 * np.pi / 180
	get_point = False
	grab_counter = 0
	theta_bound = np.array([[-75, 55], [-130, -10]])
	point_l = 5
	poppy = from_vrep('poppy.json', scene = '/home/eddiesyn/V-REP/experiment.ttt')

	def __init__(self):
		self.arm_info = np.zeros(2)
		self.EE = np.zeros(2)
		self.arm_info[0] = self.arm1_theta
		self.arm_info[1] = self.arm2_theta
		self.point_info = np.array([0.7, 0.512])
		self.point_info_init = self.point_info.copy()
		self.EE[0] = -self.arm2l*np.sin(np.sum(self.arm_info)) - self.arm1l*np.sin(self.arm_info[0])
		self.EE[1] = self.arm2l*np.cos(np.sum(self.arm_info)) + self.arm1l*np.cos(self.arm_info[0])

		# self.poppy = from_vrep('poppy.json', scene = '/home/eddiesyn/V-REP/experiment.ttt')
		for m in self.poppy.motors:
		    if m.id == 41:
		        self.motor_41 = m
		    if m.id == 44:
        		self.motor_44 = m
		self.motor_41.compliant = False
		self.motor_44.compliant = False
		self.motor_41.torque_limit = 15
		self.motor_44.torque_limit = 15
		self.motor_41.moving_speed = 10
		self.motor_44.moving_speed = 10
		move(self.motor_44, self.motor_44.present_position, -10)
		move(self.motor_41, self.motor_41.present_position, 0)


	def step(self, action):
		action_ = action * 180 / np.pi 	# action is np.array(2,)
		goal_position_1 = np.clip((self.arm_info + action_)[0], -75, 55)
		goal_position_2 = np.clip((self.arm_info + action_)[1], -130, -10)

		self.poppy.reset_simulation()

		move(self.motor_41, self.motor_41.present_position, goal_position_1)
		move(self.motor_44, self.motor_44.present_position, goal_position_2)
		self.arm_info[0] = self.motor_41.present_position 
		self.arm_info[1] = self.motor_44.present_position 
		self.EE[0] = -self.arm2l*np.sin(np.sum(self.arm_info)) - self.arm1l*np.sin(self.arm_info[0])
		self.EE[1] = self.arm2l*np.cos(np.sum(self.arm_info)) + self.arm1l*np.cos(self.arm_info[0])

		s = self.get_state()
		r = self._r_func(s[2])

		return s, r 

	def _r_func(self, distance):
		t = 50
		abs_distance = distance
		r = -abs_distance/200
		if abs_distance < self.point_l and (not self.get_point):
			r += 1.
			self.grab_counter += 1
			if self.grab_counter > t:
				r += 10.
				self.get_point = True
		elif abs_distance > self.point_l:
			self.grab_counter = 0
			self.get_point = False

		return r

	def reset(self):
		self.poppy.reset_simulation()

		self.get_point = False
		self.grab_counter = 0
		self.arm_info[0] = -75 + 130*np.random.random()
		self.arm_info[1] = -130 + 120*np.random.random()
		move(self.motor_41, self.motor_41.present_position, self.arm_info[0])
		move(self.motor_44, self.motor_44.present_position, self.arm_info[1])
		self.arm_info[0] = self.motor_41.present_position 
		self.arm_info[1] = self.motor_44.present_position 

		s_ = self.get_state()

		return s_

	def get_state(self):
		state_ = np.zeros(4)
		state_[:2] = self.arm_info
		state_[2] = np.linalg.norm(self.point_info - self.EE)
		state_[3] = 1 if self.grab_counter > 0 else 0
		return state_ 

	# def sample_action(self):
	# 	

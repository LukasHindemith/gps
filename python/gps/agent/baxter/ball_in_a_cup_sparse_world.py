import numpy as np
import rospy
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, SPARSE_COST
import baxter_interface
from baxter_interface import CHECK_VERSION

from imu_ros.srv import *
from sensor_msgs.msg import Range


class BIACSparse_World():

    def __init__(self, hyperparams):

        # create our limb instance
        self._limb = baxter_interface.Limb(hyperparams['limb'])
        #self._limb.set_joint_position_speed(0.1)
        self.x0 = hyperparams['x0']

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._start_angles = dict()

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Baxter arm running... ")

        self._rate = rospy.Rate(hyperparams['rate'])

    def run_next(self, action):
        """
        Moves forward in time one step. Makes an action
        :param action:
        """
        cmd = dict()
        for i, joint in enumerate(self._limb.joint_names()):
            cmd[joint] = action[i]
        self._limb.set_joint_velocities(cmd)
        #self._limb.set_joint_torques(cmd)
        self._rate.sleep()

    def reset_arm(self):
        """
        Moves the arm to its initial state
        """
        cmdAngle = dict()
        cmdTorque = dict()
        for i, joint in enumerate(self._limb.joint_names()):
            #cmd[joint] = self.x0[0][i]
            cmdAngle[joint] = self.x0[i]
            cmdTorque[joint] = 0.0

        self._limb.set_joint_torques(cmdTorque)
        self._limb.move_to_joint_positions(cmdAngle)
        self._rate.sleep()

    def get_state(self):
        """
        Retrieves the current state of the arm
        :return:
        """
        current_angles = np.fromiter(self._limb.joint_angles().values(), dtype=float)
        current_velocities = np.fromiter(self._limb.joint_velocities().values(), dtype=float)
        tmp_ee_pose = self._limb.endpoint_pose()['position']
        current_eepoints = np.array([tmp_ee_pose.x, tmp_ee_pose.y, tmp_ee_pose.z])

        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  #END_EFFECTOR_POINTS: current_eepoints,
                  SPARSE_COST: np.array([0])}
        return state


    def get_final_state(self):
        """
        Retrieves the current state of the arm
        :return:
        """
        current_angles = np.fromiter(self._limb.joint_angles().values(), dtype=float)
        current_velocities = np.fromiter(self._limb.joint_velocities().values(), dtype=float)
        tmp_ee_pose = self._limb.endpoint_pose()['position']
        current_eepoints = np.array([tmp_ee_pose.x, tmp_ee_pose.y, tmp_ee_pose.z])

        reward = input("give reward between 1-5 stars: ")
        while reward == "":
            reward = input("give reward between 1-5 stars: ")
        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  #END_EFFECTOR_POINTS: current_eepoints,
                  SPARSE_COST: np.array([float(reward)])}

        return state

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")

        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

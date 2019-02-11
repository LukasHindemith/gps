import numpy as np
import rospy
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, SPARSE_COST
import baxter_interface
from baxter_interface import CHECK_VERSION

class BIACSparse_World():
    """ Baxter World to solve the Ball-in-a-cup game. """

    def __init__(self, hyperparams):

        # create a limb instance
        self._limb = baxter_interface.Limb(hyperparams['limb'])
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
        Moves forward in time one step. Makes an action.
        :param action: Joint velocities
        """
        cmd = dict()
        for i, joint in enumerate(self._limb.joint_names()):
            cmd[joint] = action[i]
        self._limb.set_joint_velocities(cmd)
        self._rate.sleep()

    def reset_arm(self):
        """
        Moves the arm to its initial state.
        """
        cmdAngle = dict()
        cmdTorque = dict()
        for i, joint in enumerate(self._limb.joint_names()):
            cmdAngle[joint] = self.x0[i]
            cmdTorque[joint] = 0.0

        self._limb.set_joint_torques(cmdTorque)
        self._limb.move_to_joint_positions(cmdAngle)
        self._rate.sleep()

    def get_state(self):
        """
        Retrieves the current state of the arm.
        :return: the current state
        """
        current_angles = np.fromiter(self._limb.joint_angles().values(), dtype=float)
        current_velocities = np.fromiter(self._limb.joint_velocities().values(), dtype=float)

        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  SPARSE_COST: np.array([0])}
        return state


    def get_final_state(self):
        """
        Retrieves the current state of the arm
        :return:
        """
        current_angles = np.fromiter(self._limb.joint_angles().values(), dtype=float)
        current_velocities = np.fromiter(self._limb.joint_velocities().values(), dtype=float)

        reward = input("give reward between 1-5 stars: ")
        while reward == "":
            reward = input("give reward between 1-5 stars: ")
        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  SPARSE_COST: np.array([float(reward)])}

        return state

    def clean_shutdown(self):
        """
        Switches out of joint velocity mode to exit cleanly
        """
        print("\nExiting example...")

        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

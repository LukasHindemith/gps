import numpy as np
import rospy
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, DOOR_ANGLE, IR_RANGE, END_EFFECTOR_POINT_JACOBIANS
import baxter_interface
from baxter_interface import CHECK_VERSION

from imu_ros.srv import *
from sensor_msgs.msg import Range


class CloseCabinetWorld():

    def __init__(self, hyperparams):

        # create our limb instance
        self._limb = baxter_interface.Limb(hyperparams['limb'])
        self._limb.set_joint_position_speed(0.1)
        self.x0 = hyperparams['x0']
        self.openedAngle = abs(hyperparams['openedAngle'])
        self.closedAngle = abs(hyperparams['closedAngle'])

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

        # Calculate Angle-state
        door_angle = (abs(self.getIMUAngle())-self.closedAngle)/(self.openedAngle-self.closedAngle)
        door_angle = np.clip(door_angle, 0.0, 1.0)

        # Calculate Range-state
        minRange = 0.004
        maxRange = 0.4
        msg = rospy.wait_for_message("/robot/range/{}_hand_range/state".format(self._limb.name), Range)
        range = msg.range
        range = np.clip(range, minRange, maxRange)
        # Normalize to 0-1
        range = (range - minRange) / (maxRange - minRange)

        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  END_EFFECTOR_POINTS: current_eepoints,
                  DOOR_ANGLE: np.array([door_angle]),
                  IR_RANGE: np.array([range])}
        return state

    def getIMUAngle(self):
        rospy.wait_for_service('get_angle')
        try:
            angleClient = rospy.ServiceProxy('get_angle', GetAngle)
            data = angleClient()
            return data.angle
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        print("\nExiting example...")

        self._limb.exit_control_mode()
        if not self._init_state and self._rs.state().enabled:
            print("Disabling robot...")
            self._rs.disable()

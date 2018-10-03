import numpy as np
import rospy
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, DOOR_ANGLE, IR_RANGE, END_EFFECTOR_POINT_JACOBIANS
import baxter_interface
from baxter_interface import CHECK_VERSION

from baxter_pykdl import baxter_kinematics

from imu_ros.srv import *
from sensor_msgs.msg import Range


class ShapesorterWorld():

    def __init__(self, hyperparams):

        # create our limb instance
        self._limb = baxter_interface.Limb(hyperparams['limb'])
        #self._limb.set_joint_position_speed(0.1)
        self.x0 = hyperparams['x0']

        self.kin = baxter_kinematics(hyperparams['limb'])

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
        tmp_ee_pos = self._limb.endpoint_pose()['position']
        tmp_ee_orient = self._limb.endpoint_pose()['orientation']
        current_eepoints = np.array([tmp_ee_pos.x, tmp_ee_pos.y, tmp_ee_pos.z, tmp_ee_orient.x, tmp_ee_orient.y, tmp_ee_orient.z, tmp_ee_orient.w])

        jacobian = self.kin.jacobian()

        state = { JOINT_ANGLES: current_angles,
                  JOINT_VELOCITIES: current_velocities,
                  END_EFFECTOR_POINTS: current_eepoints,
                  END_EFFECTOR_POINT_JACOBIANS: jacobian,}
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

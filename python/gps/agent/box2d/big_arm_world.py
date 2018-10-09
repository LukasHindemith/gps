""" This file defines an environment for the Box2D 2 Link Arm simulator. """
import Box2D as b2
import numpy as np
from gps.agent.box2d.framework import Framework
from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS

class BigArmWorld(Framework):
    """ This class defines the 7 Link Arm and its environment."""
    name = "7 Link Arm"
    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(BigArmWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        self.world.gravity = (0.0, 0.0)

        fixture_length = 2
        self.x0 = x0

        rectangle_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(.5, fixture_length)),
            density=.5,
            friction=1,
        )
        square_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1,
        )
        self.base = self.world.CreateBody(
            position=(0, 15),
            fixtures=square_fixture,
        )

        self.body1 = self.world.CreateDynamicBody(
            position=(0, 2),
            fixtures=rectangle_fixture,
            angle=b2.b2_pi,
        )

        self.body2 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.body3 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.body4 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.body5 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.body6 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.body7 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 2),
            angle=b2.b2_pi,
        )

        self.target1 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target2 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target3= self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target4 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target5 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target6 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.target7 = self.world.CreateDynamicBody(
            fixtures=rectangle_fixture,
            position=(0, 0),
            angle=b2.b2_pi,
        )

        self.joint1 = self.world.CreateRevoluteJoint(
            bodyA=self.base,
            bodyB=self.body1,
            localAnchorA=(0, 0),
            localAnchorB=(0, fixture_length),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint2 = self.world.CreateRevoluteJoint(
            bodyA=self.body1,
            bodyB=self.body2,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint3 = self.world.CreateRevoluteJoint(
            bodyA=self.body2,
            bodyB=self.body3,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint4 = self.world.CreateRevoluteJoint(
            bodyA=self.body3,
            bodyB=self.body4,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint5 = self.world.CreateRevoluteJoint(
            bodyA=self.body4,
            bodyB=self.body5,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint6 = self.world.CreateRevoluteJoint(
            bodyA=self.body5,
            bodyB=self.body6,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.joint7 = self.world.CreateRevoluteJoint(
            bodyA=self.body6,
            bodyB=self.body7,
            localAnchorA=(0, -(fixture_length - 0.5)),
            localAnchorB=(0, fixture_length - 0.5),
            enableMotor=True,
            maxMotorTorque=400,
            enableLimit=False,
        )

        self.set_joint_angles(self.body1, self.body2, self.body3, self.body4, self.body5, self.body6, self.body7,
                              x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6])

        self.set_joint_angles(self.target1, self.target2, self.target3, self.target4, self.target5, self.target6, self.target7,
                              target[0], target[1], target[2], target[3], target[4], target[5], target[6])

        self.target1.active = False
        self.target2.active = False
        self.target3.active = False
        self.target4.active = False
        self.target5.active = False
        self.target6.active = False
        self.target7.active = False

        self.joint1.motorSpeed = x0[7]
        self.joint2.motorSpeed = x0[8]
        self.joint3.motorSpeed = x0[9]
        self.joint4.motorSpeed = x0[10]
        self.joint5.motorSpeed = x0[11]
        self.joint6.motorSpeed = x0[12]
        self.joint7.motorSpeed = x0[13]

    def set_joint_angles(self, body1, body2, body3, body4, body5, body6, body7,
                         angle1, angle2, angle3, angle4, angle5, angle6, angle7):
        """ Converts the given absolute angle of the arms to joint angles"""
        pos = self.base.GetWorldPoint((0, 0))
        body1.angle = angle1 + np.pi
        new_pos = body1.GetWorldPoint((0, 2))
        body1.position += pos - new_pos

        pos = body1.GetWorldPoint((0, -1.5))
        body2.angle = angle2 + body1.angle
        new_pos = body2.GetWorldPoint((0, 1.5))
        body2.position += pos - new_pos

        pos = body2.GetWorldPoint((0, -1.5))
        body3.angle = angle3 + body2.angle
        new_pos = body3.GetWorldPoint((0, 1.5))
        body3.position += pos - new_pos

        pos = body3.GetWorldPoint((0, -1.5))
        body4.angle = angle4 + body3.angle
        new_pos = body4.GetWorldPoint((0, 1.5))
        body4.position += pos - new_pos

        pos = body4.GetWorldPoint((0, -1.5))
        body5.angle = angle5 + body4.angle
        new_pos = body5.GetWorldPoint((0, 1.5))
        body5.position += pos - new_pos

        pos = body5.GetWorldPoint((0, -1.5))
        body6.angle = angle6 + body5.angle
        new_pos = body6.GetWorldPoint((0, 1.5))
        body6.position += pos - new_pos

        pos = body6.GetWorldPoint((0, -1.5))
        body7.angle = angle7 + body6.angle
        new_pos = body7.GetWorldPoint((0, 1.5))
        body7.position += pos - new_pos

    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(BigArmWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(BigArmWorld, self).run_next(action)
        else:
            if action is not None:
                self.joint1.motorSpeed = action[0]
                self.joint2.motorSpeed = action[1]
                self.joint3.motorSpeed = action[2]
                self.joint4.motorSpeed = action[3]
                self.joint5.motorSpeed = action[4]
                self.joint6.motorSpeed = action[5]
                self.joint7.motorSpeed = action[6]
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Moves forward in time one step. Called by the renderer"""
        self.joint1.motorSpeed = action[0]
        self.joint2.motorSpeed = action[1]
        self.joint3.motorSpeed = action[2]
        self.joint4.motorSpeed = action[3]
        self.joint5.motorSpeed = action[4]
        self.joint6.motorSpeed = action[5]
        self.joint7.motorSpeed = action[6]

        super(BigArmWorld, self).Step(settings)

    def reset_world(self):
        """Returns the world to its intial state"""
        self.world.ClearForces()
        self.joint1.motorSpeed = 0
        self.joint2.motorSpeed = 0
        self.joint3.motorSpeed = 0
        self.joint4.motorSpeed = 0
        self.joint5.motorSpeed = 0
        self.joint6.motorSpeed = 0
        self.joint7.motorSpeed = 0

        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.body3.linearVelocity = (0, 0)
        self.body3.angularVelocity = 0
        self.body4.linearVelocity = (0, 0)
        self.body4.angularVelocity = 0
        self.body5.linearVelocity = (0, 0)
        self.body5.angularVelocity = 0
        self.body6.linearVelocity = (0, 0)
        self.body6.angularVelocity = 0
        self.body7.linearVelocity = (0, 0)
        self.body7.angularVelocity = 0
        self.set_joint_angles(self.body1, self.body2, self.body3, self.body4, self.body5, self.body6, self.body7,
                              self.x0[0], self.x0[1], self.x0[2], self.x0[3], self.x0[4], self.x0[5], self.x0[6])

    def get_state(self):
        """Retrieves the state of the point mass"""
        state = {JOINT_ANGLES: np.array([self.joint1.angle,
                                         self.joint2.angle,
                                         self.joint3.angle,
                                         self.joint4.angle,
                                         self.joint5.angle,
                                         self.joint6.angle,
                                         self.joint7.angle]),
                 JOINT_VELOCITIES: np.array([self.joint1.speed,
                                             self.joint2.speed,
                                             self.joint3.speed,
                                             self.joint4.speed,
                                             self.joint5.speed,
                                             self.joint6.speed,
                                             self.joint7.speed]),
                 END_EFFECTOR_POINTS: np.append(np.array(self.body7.position),[0])}
        return state


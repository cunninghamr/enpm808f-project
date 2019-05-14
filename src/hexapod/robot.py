import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    from src import vrep
    from src import vrepConst
except:
    raise Exception('"vrep.py" could not be imported. This means very probably that'
                    + ' either "vrep.py" or the remoteApi library could not be found.'
                    + ' Make sure both are in the same folder as this file,'
                    + ' or appropriately adjust the file "vrep.py"')


class ObjectHandleNotFoundError(Exception):
    def __init__(self, handle_name):
        self.handle_name = handle_name

    def __str__(self):
        return 'Failed getting object handle for "{}"'.format(self.handle_name)


class Robot:
    _num_legs = 6
    _num_joints_per_leg = 3

    STATE_DIM = _num_legs * _num_joints_per_leg + 3  # servo positions and body orientation
    ACTION_DIM = _num_legs * _num_joints_per_leg  # movement of servos
    ACTION_BOUND = 0.25

    def __init__(self, client_id, model_name):
        logger.info('Created')

        self._client_id = client_id

        self.model_name = model_name

        # object handles
        self._model = None
        self._body = None
        self._joints = None

        # states
        self.walk_height = 0
        self.position = None
        self.orientation = None
        self.configuration = None

        self.reset()

    def _load(self):
        vrep.simxSetFloatSignal(self._client_id, 'walkHeight', 0, vrepConst.simx_opmode_oneshot)  # np.random.randint(6) * 0.01 - 0.03, vrepConst.simx_opmode_oneshot)

        if not self._model:
            logger.info('Loading Model')

            # random starting orientation
            z = np.random.uniform(-30, 30)
            # print('Random Starting Angle {}'.format(z))
            vrep.simxSetFloatSignal(self._client_id, 'walkAngle', z, vrepConst.simx_opmode_oneshot)

            res, self._model = vrep.simxLoadModel(self._client_id, os.path.join('src', 'vrep', '{}.ttm'.format(self.model_name)), 1,
                                                  vrepConst.simx_opmode_blocking)
            if res == vrepConst.simx_return_ok:
                logger.info('Loaded Model {} {}'.format(self.model_name, self._model))

                vrep.simxSetObjectOrientation(self._client_id, self._model, -1, [0, 0, -z * np.pi / 180], vrepConst.simx_opmode_oneshot)

                # get object handles
                self._body = self._get_object_handle('hexa_body')
                self._joints = []
                for leg in range(self._num_legs):
                    for joint in range(self._num_joints_per_leg):
                        self._joints.append(self._get_object_handle('hexa_joint{}_{}'.format(joint + 1, leg)))

                # initialize streaming of properties
                vrep.simxGetObjectPosition(self._client_id, self._body, -1, vrepConst.simx_opmode_streaming)
                vrep.simxGetObjectOrientation(self._client_id, self._body, -1, vrepConst.simx_opmode_streaming)
                for joint in self._joints:
                    vrep.simxGetJointPosition(self._client_id, joint, vrepConst.simx_opmode_streaming)

                vrep.simxSynchronousTrigger(self._client_id)  # triggers the simulation to run one time step
                vrep.simxGetPingTime(self._client_id)  # after this call, the simulation step has completed and buffer is filled
            else:
                raise Exception('Failed to load model')

    def _remove(self):
        logger.info('Removing Model')

        if self._model:
            res = vrep.simxRemoveModel(self._client_id, self._model, vrepConst.simx_opmode_blocking)
            if res == vrepConst.simx_return_ok:
                logger.info('Removed Model')

        self._model = None

    def reset(self):
        logger.info('Resetting')

        vrep.simxSetFloatSignal(self._client_id, 'walkHeight', self.walk_height, vrepConst.simx_opmode_oneshot)

        self._remove()
        self._load()

        self.sense()

    def sense(self):
        logger.info('Sensing')

        _, self.position = vrep.simxGetObjectPosition(self._client_id, self._body, -1, vrepConst.simx_opmode_buffer)
        _, self.orientation = vrep.simxGetObjectOrientation(self._client_id, self._body, -1,
                                                            vrepConst.simx_opmode_buffer)

        self.configuration = []
        for i in range(len(self._joints)):
            _, joint_position = vrep.simxGetJointPosition(self._client_id, self._joints[i], vrepConst.simx_opmode_buffer)
            self.configuration.append(joint_position)

    def act(self, action):
        logger.debug('Acting')

        # clip action
        action = np.clip(action, -self.ACTION_BOUND, self.ACTION_BOUND)

        for i in range(len(action)):
            vrep.simxSetJointTargetPosition(self._client_id, self._joints[i], self.configuration[i] + action[i],
                                            vrepConst.simx_opmode_oneshot)

    def get_state(self):
        return np.append(self.configuration, self.orientation)

    def _get_object_handle(self, handle_name):
        res, handle = vrep.simxGetObjectHandle(self._client_id, handle_name, vrepConst.simx_opmode_blocking)
        if res != vrepConst.simx_return_ok:
            raise ObjectHandleNotFoundError(handle_name)
        return handle

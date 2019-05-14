import logging
import os
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

try:
    from src import vrep
    from src import vrepConst
except:
    logger.error('"vrep.py" could not be imported. This means very probably that'
                 + ' either "vrep.py" or the remoteApi library could not be found.'
                 + ' Make sure both are in the same folder as this file,'
                 + ' or appropriately adjust the file "vrep.py"')


class Env:
    def __init__(self, scene, render):
        self._render = render

        # close all open connections and connect to vrep
        vrep.simxFinish(-1)
        client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

        if client_id == -1:
            logger.error('Error connecting to vrep')
        else:
            logger.info('Connected to remote API server')

            self.client_id = client_id

            # load the scene into vrep
            res = vrep.simxLoadScene(self.client_id, os.path.join('src', 'vrep', '{}.ttt'.format(scene)), 0xFF, vrepConst.simx_opmode_blocking)
            if res == vrepConst.simx_return_ok:
                logger.info('Loaded scene into vrep')
            else:
                raise Exception('Failed loading scene into vrep.')

            vrep.simxGetFloatSignal(self.client_id, 'mySimulationTime', vrepConst.simx_opmode_streaming)

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        # run the scene synchronously
        vrep.simxSynchronous(self.client_id, True)

        # disable the simulation display in vrep to increase simulation speed
        if not self._render:
            vrep.simxSetBooleanParameter(self.client_id, vrepConst.sim_boolparam_display_enabled, False,
                                         vrepConst.simx_opmode_oneshot)

        # start the simulation
        res = vrep.simxStartSimulation(self.client_id, vrepConst.simx_opmode_blocking)
        if res == vrepConst.simx_return_ok:
            logger.info('Started simulation.')
        else:
            raise Exception('Failed starting simulation.')

        # self.step()

    def step(self):
        logging.debug('Step')

        vrep.simxSynchronousTrigger(self.client_id)  # triggers the simulation to run one time step
        vrep.simxGetPingTime(self.client_id)  # after this call, the simulation step has completed

        if logger.level <= logging.DEBUG:
            t = vrep.simxGetFloatSignal(self.client_id, 'mySimulationTime', vrepConst.simx_opmode_blocking)
            logger.info('Simulation Time {}'.format(t))

    def stop(self):
        # stop the simulation if running
        res = vrep.simxStopSimulation(self.client_id, vrepConst.simx_opmode_blocking)
        if res == vrepConst.simx_return_ok:
            logger.info('Stopped simulation')
        else:
            raise Exception('Failed stopping simulation')

    def reset(self):
        logger.info('Resetting')

        self.stop()

        # necessary for vrep to finish stopping
        for _ in range(10):
            vrep.simxSynchronousTrigger(self.client_id)

        time.sleep(1)

        self.start()


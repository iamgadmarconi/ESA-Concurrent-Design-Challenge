import os

import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
# import message declarations
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D
# import FSW Algorithm related support
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
# import simulation related support
from Basilisk.simulation import spacecraft
# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
# attempt to import vizard
from Basilisk.utilities import vizSupport
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError,
                                    inertial3D, rwMotorTorque, rwMotorVoltage)
from Basilisk.simulation import reactionWheelStateEffector, motorVoltageInterface, simpleNav, spacecraft
from Basilisk.utilities import (SimulationBaseClass, fswSetupRW, macros,
                                orbitalMotion, simIncludeGravBody,
                                simIncludeRW, unitTestSupport, vizSupport)
from cases.angular_acceleration_util import calculate_and_print_angular_acceleration

from cases.spacecraft import Spacecraft

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Create sinusoidal torque parameters
_SINE_AMPLITUDE = 0.003  # N-m
_SINE_FREQUENCY = 1/(12*3600)  # Hz (period of 12 hours)
_LAST_CALL_TIME = 0.0  # Track last call time
_CALL_COUNT = 0  # Track number of calls
_CURRENT_TORQUE = 0.0  # Track current torque value

class SinusoidalTorque(sysModel.SysModel):
    def __init__(self):
        super(SinusoidalTorque, self).__init__()
        
        # Initialize internal tracking values
        self.amplitude = _SINE_AMPLITUDE
        self.frequency = _SINE_FREQUENCY
        self.lastCallTime = 0.0
        
        # Add a bias to shift the sine curve up by 300 micro Nm
        self.bias = 10e-5  # Nm
        
        # Create external torque output message
        self.torqueOutMsg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        """
        Reset method initializes module variables to their starting values
        :param CurrentSimNanos: current simulation time in nano-seconds
        :return: none
        """
        global _CURRENT_TORQUE, _CALL_COUNT, _LAST_CALL_TIME
        
        # Reset tracking variables
        _CURRENT_TORQUE = 0.0
        _CALL_COUNT = 0
        _LAST_CALL_TIME = 0.0
        
        # Initialize output message with bias torque (not zero)
        torquePayload = messaging.CmdTorqueBodyMsgPayload()
        # Use flat list format instead of nested lists
        torquePayload.torqueRequestBody = [0.0, 0.0, self.bias]
        self.torqueOutMsg.write(torquePayload, CurrentSimNanos, self.moduleID)
        
        # Log reset event
        self.bskLogger.bskLog(
            sysModel.BSK_INFORMATION, 
            f"Reset SinusoidalTorque module at {CurrentSimNanos*1e-9}s with bias {self.bias} Nm"
        )

    def UpdateState(self, CurrentSimNanos):
        """
        UpdateState method computes the sinusoidal torque based on the current time
        :param CurrentSimNanos: current simulation time in nano-seconds
        :return: none
        """
        global _CURRENT_TORQUE, _CALL_COUNT, _LAST_CALL_TIME
        
        # Convert time to seconds for easier calculations
        currentTime = CurrentSimNanos * 1e-9
        
        # Calculate sinusoidal torque value with bias
        # Formula: amplitude * sin(2π * frequency * time) + bias
        sineTorque = self.amplitude * np.sin(2.0 * np.pi * self.frequency * currentTime)
        torqueValue = sineTorque + self.bias
        
        # Create output message with computed torque
        torquePayload = messaging.CmdTorqueBodyMsgPayload()
        # Apply torque to Z-axis using flat list format
        torquePayload.torqueRequestBody = [0.0, 0.0, torqueValue]
        self.torqueOutMsg.write(torquePayload, CurrentSimNanos, self.moduleID)
        
        # Update tracking variables
        _CURRENT_TORQUE = torqueValue
        _LAST_CALL_TIME = currentTime
        _CALL_COUNT += 1
        
        # Periodically log the torque value (every 10th call to avoid excessive logging)
        if _CALL_COUNT % 10000 == 0:
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION, 
                f"SinusoidalTorque: t={currentTime:.1f}s, value={torqueValue:.6f}Nm (sin: {sineTorque:.6f} + bias: {self.bias:.6f})"
            )

def run(show_plots, useUnmodeledTorque, useIntGain, useKnownTorque):
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots
        useUnmodeledTorque (bool): Specify if an external torque should be included
        useIntGain (bool): Specify if the feedback control uses an integral feedback term
        useKnownTorque (bool): Specify if the external torque is feed forward in the contro

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(True)
    # set the simulation time variable used later on
    simulationTime = macros.min2nano(14400.)

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(1.)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #

    # initialize spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bsk-Sat"
    # define the simulation inertia
    I = Spacecraft.inertia
    scObject.hub.mHub = Spacecraft.mass  # kg - spacecraft mass
    scObject.hub.r_BcB_B = Spacecraft.r_BcB_B  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject)

    # setup extForceTorque module
    # the control torque is read in through the messaging system
    if useUnmodeledTorque:
        # Set sinusoidal parameters to match the SRP torque from image
        global _SINE_AMPLITUDE, _SINE_FREQUENCY
        # Set this BEFORE creating the object
        _SINE_AMPLITUDE = 0.003  # Nm for the sinusoidal torque
        _SINE_FREQUENCY = 1/(12*3600)  # Period of ~12 hours converted to Hz
        
        print(f"Creating SinusoidalTorque with amplitude: {_SINE_AMPLITUDE} Nm, frequency: {_SINE_FREQUENCY} Hz")
        
        # Create sinusoidal torque Python module
        sineTorqueModule = SinusoidalTorque()
        sineTorqueModule.ModelTag = "SRPSinusoidalTorque"
        scSim.AddModelToTask(simTaskName, sineTorqueModule)
        
        # Create extForceTorque module to apply the torque
        extFTObject = extForceTorque.ExtForceTorque()
        extFTObject.ModelTag = "SRPDisturbance"
        
        # Connect the sinusoidal torque to the external force module
        extFTObject.cmdTorqueInMsg.subscribeTo(sineTorqueModule.torqueOutMsg)
    else:
        extFTObject = extForceTorque.ExtForceTorque()
        extFTObject.extTorquePntB_B = [[0.003], [0.0], [0.0]]
        extFTObject.ModelTag = "externalDisturbance"
        
    # Debug output to verify the object exists
    print(f"ExtForceTorque object: {extFTObject}")
    
    # Add the dynamic effector to the spacecraft
    print("Adding torque object to spacecraft...")
    scObject.addDynamicEffector(extFTObject)
    
    # Add the model to the simulation task
    print("Adding torque object to simulation task...")
    scSim.AddModelToTask(simTaskName, extFTObject)

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"

    pos_sigma_sc = 0.5       # Position accuracy (m) - less critical for pure ADCS
    vel_sigma_sc = 0.005     # Velocity accuracy (m/s)
    att_sigma_sc = 0.00004   # Attitude knowledge accuracy (rad) - ~0.0023° from star tracker spec
    rate_sigma_sc = 0.00001  # Angular rate knowledge accuracy (rad/s) - from Astrix-120's 5e-4°/hr
    sun_sigma_sc = 0.0003    # Sun vector knowledge accuracy - from fine sun sensor 0.01-0.05° spec
    dv_sigma_sc = 0.0005     # Delta-v accuracy (m/s)
    p_matrix_sc = np.diag([pos_sigma_sc, pos_sigma_sc, pos_sigma_sc,
                           vel_sigma_sc, vel_sigma_sc, vel_sigma_sc,
                           att_sigma_sc, att_sigma_sc, att_sigma_sc,
                           rate_sigma_sc, rate_sigma_sc, rate_sigma_sc,
                           sun_sigma_sc, sun_sigma_sc, sun_sigma_sc,
                           dv_sigma_sc, dv_sigma_sc, dv_sigma_sc])
    walk_bounds_sc = [[0.5], [0.5], [0.5],     # Position bounds (m) 
                      [0.005], [0.005], [0.005], # Velocity bounds (m/s)
                      [0.00005], [0.00005], [0.00005], # Attitude bounds (rad) - star tracker performance
                      [0.00001], [0.00001], [0.00001], # Rate bounds (rad/s) - Astrix-120 performance
                      [0.0005], [0.0005], [0.0005],   # Sun vector - fine sun sensor performance
                      [0.0005], [0.0005], [0.0005]]   # Delta-v bounds
    sNavObject.PMatrix = p_matrix_sc
    sNavObject.walkBounds = walk_bounds_sc

    scSim.AddModelToTask(simTaskName, sNavObject)
    # Make the RW control all three body axes
    controlAxes_B = [1, 0, 0, 
                    0, 1, 0, 
                    0, 0, 1]

    rwFactory = Spacecraft.rwFactory

    RW1 = Spacecraft.RW1
    RW2 = Spacecraft.RW2
    RW3 = Spacecraft.RW3
    RW4 = Spacecraft.RW4
    RW = [RW1, RW2, RW3, RW4]
    numRW = rwFactory.getNumOfDevices()

    # create RW object container and tie to spacecraft object
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)

    # add RW object array to the simulation process.  This is required for the UpdateState() method
    # to be called which logs the RW states
    scSim.AddModelToTask(simTaskName, rwStateEffector)

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    rwMotorTorqueObj.controlAxes_B = controlAxes_B
    scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)
    #
    #   setup the FSW algorithm tasks
    #

    # setup inertial3D guidance module
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.]  # set the desired inertial orientation

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)

    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl)
    mrpControl.K = 30
    if useIntGain:
        mrpControl.Ki = 0.0002  # make value negative to turn off integral feedback
    else:
        mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 3000.0
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1
    if useKnownTorque:
        mrpControl.knownTorquePntB_B = [0.005, -0.005, 0.005]

    #
    #   Setup data logging before the simulation is initialized
    #
    numDataPoints = 1000
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    mrpLog = mrpControl.cmdTorqueOutMsg.recorder(samplingTime)
    mrpLogRW = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
    # Add navigation attitude log
    snAttLog = sNavObject.attOutMsg.recorder(samplingTime)
    # Log sun safe point guidance
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, mrpLog)
    scSim.AddModelToTask(simTaskName, mrpLogRW)
    scSim.AddModelToTask(simTaskName, snAttLog)
    rwLogs = []
    for item in range(numRW):
        rwLogs.append(rwStateEffector.rwOutMsgs[item].recorder(samplingTime))
        scSim.AddModelToTask(simTaskName, rwLogs[item])
    #
    # create simulation messages
    #

    # create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
    configDataMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    fswRwParamMsg = rwFactory.getConfigMessage()

    # The primary difference is that the gravity body is not included.
    # When initializing the spacecraft states, only the attitude states must be set.  The position and velocity
    # states are initialized automatically to zero.
    scObject.hub.sigma_BNInit = [[0.0000], [0.0000], [-0.0000]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # rad/s - omega_BN_B

    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                              , modelDictionaryKeyList="3USat"
                                              # , saveFile=fileName
                                              )
    #
    # connect the messages to the modules
    #
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    # link the RW motor torque module to the MRP Feedback control module
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    dataLr = mrpLog.torqueRequestBody
    dataOmegaRW = mrpLogRW.wheelSpeeds
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    timeAxis = attErrorLog.times()
    np.set_printoptions(precision=16)

    #
    #   plot the results
    #
    # plt.close("all")  # clears out plots from earlier test runs
    # plt.figure(1)
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2MIN, dataSigmaBR[:, idx],
    #              color=unitTestSupport.getLineColor(idx, 3),
    #              label=r'$\sigma_' + str(idx) + '$')
    # plt.legend(loc='lower right')
    # plt.xlabel('Time [min]')
    # plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
    # figureList = {}
    # pltName = fileName + "1" + str(int(useUnmodeledTorque)) + str(int(useIntGain))+ str(int(useKnownTorque))
    # figureList[pltName] = plt.figure(1)

    # plt.figure(2)
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2MIN, dataLr[:, idx],
    #              color=unitTestSupport.getLineColor(idx, 3),
    #              label='$L_{r,' + str(idx) + '}$')
    # plt.legend(loc='lower right')
    # plt.xlabel('Time [min]')
    # plt.ylabel('Control Torque $L_r$ [Nm]')
    # pltName = fileName + "2" + str(int(useUnmodeledTorque)) + str(int(useIntGain)) + str(int(useKnownTorque))
    # figureList[pltName] = plt.figure(2)

    # plt.figure(3)
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2MIN, dataOmegaBR[:, idx],
    #              color=unitTestSupport.getLineColor(idx, 3),
    #              label=r'$\omega_{BR,' + str(idx) + '}$')
    # plt.legend(loc='lower right')
    # plt.xlabel('Time [min]')
    # plt.ylabel('Rate Tracking Error [rad/s] ')

    plot_track_error_norm(timeAxis*macros.NANO2MIN, dataSigmaBR, fontsize=16)
    plot_rate_error_norm(timeAxis*macros.NANO2MIN, dataOmegaBR, fontsize=16)
    plot_rw_momenta(timeAxis*macros.NANO2MIN, dataOmegaRW, RW, numRW, fontsize=16)
    plot_rw_speeds(timeAxis*macros.NANO2MIN, dataOmegaRW, numRW, fontsize=16)

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    # Calculate and print angular acceleration
    calculate_and_print_angular_acceleration(snAttLog)
    
    # If we're using the unmodeled torque, let's verify it was applied
    if useUnmodeledTorque and isinstance(extFTObject, extForceTorque.ExtForceTorque):
        print("\n=== TORQUE VERIFICATION ===")
        print(f"Final extTorquePntB_B value: {extFTObject.extTorquePntB_B}")
        print(f"Final torqueExternalPntB_B value: {extFTObject.torqueExternalPntB_B}")
        
        # Check if angular acceleration is consistent with applied torque
        accs = calculate_and_print_angular_acceleration(snAttLog)
        
        # Get the max acceleration value (scalar)
        maxAcc = accs[3]  # This is the magnitude value
        
        # Calculate expected acceleration - torque/inertia
        # Get the z-axis moment of inertia
        I_zz = Spacecraft.inertia[2, 2] if hasattr(Spacecraft.inertia, 'shape') else 0.25e5
        expectedAcc = _SINE_AMPLITUDE / I_zz  # Rough estimate for z-axis
        
        print(f"Max angular acceleration: {float(maxAcc):.6f} rad/s²")
        print(f"Amplitude: {_SINE_AMPLITUDE} Nm, I_zz: {I_zz} kg·m²")
        print(f"Expected max acceleration: ~{expectedAcc:.6f} rad/s²")
        print(f"Current torque value: {_CURRENT_TORQUE:.6f} Nm")

def plot_rw_speeds(timeData, dataOmegaRW, numRW, fontsize=14):
    """Plot the RW spin rates."""
    plt.figure(4)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                color=unitTestSupport.getLineColor(idx, numRW),
                label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xlabel('Time [min]', fontsize=fontsize)
    plt.ylabel('RW Speed (RPM) ', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)


def plot_track_error_norm(timeLineSet, dataSigmaBR, fontsize=14):
    """Plot the attitude tracking error norm value."""
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    vectorData = dataSigmaBR
    sNorm = np.array([np.linalg.norm(v) for v in vectorData])
    plt.plot(timeLineSet, sNorm,
             color=unitTestSupport.getLineColor(1, 3),
             )
    plt.xlabel('Time [min]', fontsize=fontsize)
    plt.ylabel(r'Attitude Error Norm $|\sigma_{B/R}|$', fontsize=fontsize)
    ax.set_yscale('log')
    plt.tick_params(labelsize=fontsize)

def plot_rate_error_norm(timeLineSet, dataOmegaBR, fontsize=14):
    """Plot the rate tracking error norm value."""
    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    vectorData = dataOmegaBR
    sNorm = np.array([np.linalg.norm(v) for v in vectorData])
    plt.plot(timeLineSet, sNorm,
             color=unitTestSupport.getLineColor(1, 3),
             )
    plt.xlabel('Time [min]', fontsize=fontsize)
    plt.ylabel(r'Rate Error Norm $|\omega_{B/R}|$', fontsize=fontsize)
    ax.set_yscale('log')


def plot_rw_momenta(timeData, dataOmegaRw, RW, numRW, fontsize=14):
    """Plot the RW momenta."""
    totMomentumNorm = []
    for j in range(len(timeData)):
        totMomentum = np.array([0,0,0])
        for idx in range(numRW):
            for k in range(3):
                totMomentum[k] = totMomentum[k] + dataOmegaRw[j, idx] * RW[idx].Js * RW[idx].gsHat_B[k][0]
        totMomentumNorm.append(np.linalg.norm(totMomentum))
    plt.figure(3)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRw[:, idx] * RW[idx].Js,
                 color=unitTestSupport.getLineColor(idx, numRW),
                 label=r'$H_{' + str(idx+1) + r'}$')
    plt.plot(timeData, totMomentumNorm, '--',
             label=r'$\|H\|$')
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.xlabel('Time [min]', fontsize=fontsize)
    plt.ylabel('RW Momentum (Nms)', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    run(
        True,  # show_plots
        True,  # useUnmodeledTorque
        False,  # useIntGain
        False  # useKnownTorque
    )

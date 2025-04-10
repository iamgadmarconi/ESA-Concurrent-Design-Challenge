import os

import matplotlib.pyplot as plt
import numpy as np
from Basilisk import __path__

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


from Basilisk.utilities import (SimulationBaseClass, macros, simIncludeGravBody, vizSupport, unitTestSupport, orbitalMotion, fswSetupThrusters, simIncludeThruster)
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav, ephemerisConverter, planetEphemeris, thrusterDynamicEffector
from Basilisk.fswAlgorithms import mrpFeedback, attTrackingError, velocityPoint, locationPointing
from Basilisk.architecture import messaging, astroConstants

try:
    from Basilisk.simulation import vizInterface
except ImportError:
    pass

# The path to the location of Basilisk
# Used to get the location of supporting data.
fileName = os.path.basename(os.path.splitext(__file__)[0])


def run(show_plots):
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    # Configure the simulation
    scSim = SimulationBaseClass.SimBaseClass()

    # Shows the simulation progress bar in the terminal
    scSim.SetProgressBar(True)

    # Create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and specify the simulation time step information
    simulationTimeStep = macros.sec2nano(10.0)

    # Add dynamics task to the simulation process
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Setup celestial object ephemeris module for the asteroid
    gravBodyEphem = planetEphemeris.PlanetEphemeris()
    gravBodyEphem.ModelTag = 'planetEphemeris'
    scSim.AddModelToTask(simTaskName, gravBodyEphem)
    gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(["320P/McNaught"]))

    # Specify orbital parameters of the asteroid
    timeInitString = "2036 January 1 0:00:00.0"
    diam = 1600  # m
    G = 6.67408 * (10 ** -11)  # m^3 / kg*s^2
    massBennu = 1.34 * (10 ** 11)  # kg
    mu = G * massBennu  # Bennu grav. parameter, m^3/s^2
    oeAsteroid = planetEphemeris.ClassicElements()
    oeAsteroid.a = 3.103 * astroConstants.AU * 1000  # m
    oeAsteroid.e = 0.6824
    oeAsteroid.i = 4.89 * macros.D2R
    oeAsteroid.Omega = 295.96 * macros.D2R
    oeAsteroid.omega = 0.67 * macros.D2R
    oeAsteroid.f = 0.0 * macros.D2R
    gravBodyEphem.planetElements = planetEphemeris.classicElementVector([oeAsteroid])

    # Specify orientation parameters of the asteroid
    gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([85.65 * macros.D2R])
    gravBodyEphem.declination = planetEphemeris.DoubleVector([-60.17 * macros.D2R])
    gravBodyEphem.lst0 = planetEphemeris.DoubleVector([0.0 * macros.D2R])
    gravBodyEphem.rotRate = planetEphemeris.DoubleVector([360 * macros.D2R / (12.296057 * 3600.)])  # rad/sec

    # Set orbital radii about asteroid
    r0 = diam/2.0 + 500  # capture orbit, meters
    r1 = diam/2.0 + 200  # meters, very close fly-by, elliptic orbit
    rP = r0
    rA = 3*rP

    # Set orbital periods
    P0 = np.pi*2/np.sqrt(mu/(r0**3))
    P01 = np.pi*2/np.sqrt(mu/(((r0+r1)/2)**3))
    P1 = np.pi*2/np.sqrt(mu/(r1**3))

    # Create additional gravitational bodies
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravFactory.createBodies("earth", "sun")

    # Set gravity body index values
    earthIdx = 0
    sunIdx = 1
    asteroidIdx = 2

    # Create and configure the default SPICE support module. The first step is to store
    # the date and time of the start of the simulation.
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)

    # Add the SPICE object to the simulation task list
    scSim.AddModelToTask(simTaskName, spiceObject)

    # Create the asteroid custom gravitational body
    asteroid = gravFactory.createCustomGravObject("320P/McNaught", mu
                                                #, modelDictionaryKey="asteroid2"
                                                , radEquator=1600. / 2.0
                                                , radiusRatio= 0.9
                                                )
    asteroid.isCentralBody = True  # ensures the asteroid is the central gravitational body
    asteroid.planetBodyInMsg.subscribeTo(gravBodyEphem.planetOutMsgs[0])  # connect asteroid ephem. to custom grav body

    # Create the spacecraft object
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bskSat"

    # Connect all gravitational bodies to the spacecraft
    gravFactory.addBodiesTo(scObject)
    scSim.AddModelToTask(simTaskName, scObject)

    # Create an ephemeris converter to convert messages of type
    # 'SpicePlanetStateMsgPayload' to 'EphemerisMsgPayload'
    ephemObject = ephemerisConverter.EphemerisConverter()
    ephemObject.ModelTag = 'EphemData'
    ephemObject.addSpiceInputMsg(spiceObject.planetStateOutMsgs[earthIdx])
    ephemObject.addSpiceInputMsg(spiceObject.planetStateOutMsgs[sunIdx])
    # Recall the asteroid was not created with Spice.
    ephemObject.addSpiceInputMsg(gravBodyEphem.planetOutMsgs[0])
    scSim.AddModelToTask(simTaskName, ephemObject)

    # Define the spacecraft inertia
    I = [900., 0., 0.,
         0., 800., 0.,
         0., 0., 600.]
    scObject.hub.mHub = 750.0  # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # Define the initial spacecraft orbit about the asteroid
    oe = orbitalMotion.ClassicElements()
    oe.a = (rP + rA)/2.0
    oe.e = 1 - (rP / oe.a)
    oe.i = 5.0 * macros.D2R
    oe.Omega = 180.0 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = -45.0 * macros.D2R
    Ecc = np.arctan(np.tan(-oe.f/2)*np.sqrt((1-oe.e)/(1+oe.e)))*2 # eccentric anomaly
    M = Ecc - oe.e*np.sin(Ecc) # mean anomaly
    n = np.sqrt(mu/(oe.a**3))
    h = np.sqrt(mu*oe.a*(1-oe.e**2)) # specific angular momentum
    vP = h/rP
    V_SC_C_B = np.sqrt(mu / rP)     # [m/s] (2) spacecraft circular parking speed relative to bennu.
    Delta_V_Parking_Orbit = V_SC_C_B - vP

    # Setting initial position and velocity vectors using orbital elements
    r_N, v_N = orbitalMotion.elem2rv(mu, oe)
    T1 = M/n  # time until spacecraft reaches periapsis of arrival trajectory

    # Initialize spacecraft states with the initialization variables
    scObject.hub.r_CN_NInit = r_N  # [m]   = r_BN_N
    scObject.hub.v_CN_NInit = v_N  # [m/s] = v_BN_N
    scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
    scObject.hub.omega_BN_BInit = [[0.000], [-0.00], [0.00]]  # rad/s - omega_BN_B

    # Set up the extForceTorque module
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "externalDisturbance"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)

    # Add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Set up asteroid-relative velocityPoint guidance module
    velAsteroidGuidance = velocityPoint.velocityPoint()
    velAsteroidGuidance.ModelTag = "velocityPointAsteroid"
    velAsteroidGuidance.transNavInMsg.subscribeTo(sNavObject.transOutMsg)
    velAsteroidGuidance.celBodyInMsg.subscribeTo(ephemObject.ephemOutMsgs[asteroidIdx])
    velAsteroidGuidance.mu = mu
    scSim.AddModelToTask(simTaskName, velAsteroidGuidance)

    # Set up sensor science-pointing guidance module
    cameraLocation = [0.0, 1.5, 0.0]
    sciencePointGuidance = locationPointing.locationPointing()
    sciencePointGuidance.ModelTag = "sciencePointAsteroid"
    sciencePointGuidance.celBodyInMsg.subscribeTo(ephemObject.ephemOutMsgs[asteroidIdx])
    sciencePointGuidance.scTransInMsg.subscribeTo(sNavObject.transOutMsg)
    sciencePointGuidance.scAttInMsg.subscribeTo(sNavObject.attOutMsg)
#    sciencePointGuidance.scTargetInMsg.subscribeTo(sNavObject.transOutMsg)
    sciencePointGuidance.pHat_B = cameraLocation  # y-axis set for science-pointing sensor
    sciencePointGuidance.useBoresightRateDamping = 1
    scSim.AddModelToTask(simTaskName, sciencePointGuidance)

    # Set up the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)
    attError.attRefInMsg.subscribeTo(sciencePointGuidance.attRefOutMsg)  # initial flight mode
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)

    length, width, height = [1.6, 1, 1.3]

    location = [
        # Front face (-Y) thrusters (1, 2)
        [-length/4, -width/2, 0],  # Thruster 1, bottom left front
        [length/4, -width/2, 0],   # Thruster 2, bottom right front

        # Front face (+Y) thrusters (3, 4)
        [-length/4, width/2, 0],    # Thruster 3, top left front
        [length/4, width/2, 0],     # Thruster 4, top right front

        # Left face (-X) thrusters (5, 6)
        [-length/2, 0, -width/4],   # Thruster 5, top left front
        [-length/2, 0, width/4],    # Thruster 6, top right front

        # Right face (+X) thrusters (7, 8)
        [length/2, 0, -width/4],   # Thruster 7, top left front
        [length/2, 0, width/4],    # Thruster 8, top right front


        # Bottom face (-Z) thrusters (9, 10)
        [0, -width/4, -height/2],   # Thruster 9, top left front
        [0, width/4, -height/2],    # Thruster 10, top right front

        # Top face (+Z) thrusters (11, 12)
        [0, -width/4, height/2],    # Thruster 11, top left front
        [0, width/4, height/2],     # Thruster 12, top right front
    ]
    

    direction = [
        [0, -1, 0],  # Thruster 1, -Y direction
        [0, -1, 0],  # Thruster 2, -Y direction
        [0, 1, 0],   # Thruster 3, +Y direction
        [0, 1, 0],   # Thruster 4, +Y direction
        [-1, 0, 0],  # Thruster 5, -X direction
        [-1, 0, 0],  # Thruster 6, -X direction
        [1, 0, 0],   # Thruster 7, +X direction
        [1, 0, 0],   # Thruster 8, +X direction
        [0, 0, -1],  # Thruster 9, -Z direction
        [0, 0, -1],  # Thruster 10, -Z direction
        [0, 0, 1],   # Thruster 11, +Z direction
        [0, 0, 1],   # Thruster 12, +Z direction
    ]

    thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
    scSim.AddModelToTask(simTaskName, thrusterSet)

    thFactory = simIncludeThruster.thrusterFactory()

    for pos_B, dir_B in zip(location, direction):
        thFactory.create('MOOG_Monarc_22_6', pos_B, dir_B)

    numTh = thFactory.getNumOfDevices()

    # create thruster object container and tie to spacecraft object
    thrModelTag = "ACSThrusterDynamics"
    thFactory.addToSpacecraft(thrModelTag, thrusterSet, scObject)

    fswSetupThrusters.clearSetup()
    for pos_B, dir_B in zip(location, direction):
        fswSetupThrusters.create(pos_B, dir_B, 1)
    fswThrConfigMsg = fswSetupThrusters.writeConfigMessage()

    # Create the FSW vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # Set up the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    mrpControl.Ki = -1.0  # make value negative to turn off integral feedback
    II = 900.
    mrpControl.P = 2*II/(20*60)
    mrpControl.K = mrpControl.P*mrpControl.P/II
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    # Connect the torque command to external torque effector
    extFTObject.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

    # Set the simulation time
    # Set up data logging before the simulation is initialized
    scRec = scObject.scStateOutMsg.recorder()
    astRec = gravBodyEphem.planetOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, scRec)
    scSim.AddModelToTask(simTaskName, astRec)

    if vizSupport.vizFound:
        # Set up the sensor for the science-pointing mode
        genericSensor = vizInterface.GenericSensor()
        genericSensor.r_SB_B = cameraLocation
        genericSensor.fieldOfView.push_back(10.0 * macros.D2R)
        genericSensor.fieldOfView.push_back(10.0 * macros.D2R)
        genericSensor.normalVector = cameraLocation
        genericSensor.size = 10
        genericSensor.color = vizInterface.IntVector(vizSupport.toRGBA255("white", alpha=0.1))
        genericSensor.label = "scienceCamera"
        genericSensor.genericSensorCmd = 1

        # Set up the thruster visualization info
        # Note: This process is different from the usual procedure of creating a thruster effector.
        # The following code creates a thruster visualization only.
        # before adding the thruster
        scData = vizInterface.VizSpacecraftData()
        scData.spacecraftName = scObject.ModelTag
        scData.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
        scData.genericSensorList = vizInterface.GenericSensorVector([genericSensor])

        thrusterMsgInfo = messaging.THROutputMsgPayload()
        thrusterMsgInfo.maxThrust = 1  # Newtons
        thrusterMsgInfo.thrustForce = 0  # Newtons
        thrusterMsgInfo.thrusterLocation = [0, 0, -1.5]
        thrusterMsgInfo.thrusterDirection = [0, 0, 1]
        thrMsg = messaging.THROutputMsg().write(thrusterMsgInfo)
        scData.thrInMsgs = messaging.THROutputMsgInMsgsVector([thrMsg.addSubscriber()])

        thrInfo = vizInterface.ThrClusterMap()
        thrInfo.thrTag = "DV"
        scData.thrInfo = vizInterface.ThrClusterVector([thrInfo])

        # Create the Vizard visualization file and set parameters
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                                  , saveFile=fileName
                                                  )
        viz.epochInMsg.subscribeTo(gravFactory.epochMsg)

        viz.settings.showCelestialBodyLabels = 1
        viz.settings.showSpacecraftLabels = 1
        viz.settings.truePathFixedFrame = "320P/McNaught"
        viz.settings.trueTrajectoryLinesOn = 5  # relative to celestial body fixed frame

        viz.settings.scViewToPlanetViewBoundaryMultiplier = 100
        viz.settings.planetViewToHelioViewBoundaryMultiplier = 100
        viz.settings.orbitLinesOn = -1
        viz.settings.keyboardAngularRate = np.deg2rad(0.5)

        # Create the science mode camera
        vizSupport.createStandardCamera(viz, setMode=1, spacecraftName=scObject.ModelTag,
                                        fieldOfView=10 * macros.D2R,
                                        displayName="10˚ FOV Camera",
                                        pointingVector_B=[0, 1, 0], position_B=cameraLocation)

        # Note: After running the enableUnityVisualization() method, we need to clear the
        # vizInterface spacecraft data container, scData, and push our custom copy to it.
        viz.scData.clear()
        viz.scData.push_back(scData)

    # Initialize and execute the simulation for the first section
    scSim.InitializeSimulation()


    def runSensorSciencePointing(simTime):
        nonlocal simulationTime
        attError.attRefInMsg.subscribeTo(sciencePointGuidance.attRefOutMsg)
        if vizSupport.vizFound:
            genericSensor.isHidden = 0
            thrusterMsgInfo.thrustForce = 0
            thrMsg.write(thrusterMsgInfo, simulationTime)
        attError.sigma_R0R = [0, 0, 0]
        simulationTime += macros.sec2nano(simTime)
        scSim.ConfigureStopTime(simulationTime)
        scSim.ExecuteSimulation()


    def runDvBurn(simTime, burnSign, planetMsg):
        nonlocal simulationTime
        attError.attRefInMsg.subscribeTo(planetMsg)
        if vizSupport.vizFound:
            genericSensor.isHidden = 1
        if burnSign > 0:
            attError.sigma_R0R = [np.tan((np.pi/2)/4), 0, 0]
        else:
            attError.sigma_R0R = [-np.tan((np.pi / 2) / 4), 0, 0]
        minTime = 40 * 60
        if simTime < minTime:
            print("ERROR: runPosDvBurn must have simTime larger than " + str(minTime) + " min")
            exit(1)
        else:
            simulationTime += macros.sec2nano(minTime)
            scSim.ConfigureStopTime(simulationTime)
            scSim.ExecuteSimulation()
            if vizSupport.vizFound:
                thrusterMsgInfo.thrustForce = thrusterMsgInfo.maxThrust
                thrMsg.write(thrusterMsgInfo, simulationTime)
            simulationTime += macros.sec2nano(simTime - minTime)
            scSim.ConfigureStopTime(simulationTime)
            scSim.ExecuteSimulation()

    simulationTime = 0
    np.set_printoptions(precision=16)
    burnTime = 200*60

    # Run thruster burn for arrival to the capture orbit with thrusters on
    runDvBurn(T1, -1, velAsteroidGuidance.attRefOutMsg)

    # Get current spacecraft states
    velRef = scObject.dynManager.getStateObject(scObject.hub.nameOfHubVelocity)
    vN = scRec.v_BN_N[-1] - astRec.VelocityVector[-1]

    # Apply a delta V and set the new velocity state in the circular capture orbit
    vHat = vN / np.linalg.norm(vN)
    vN = vN + Delta_V_Parking_Orbit * vHat
    velRef.setState(vN)

    # Travel in a circular orbit at r0, incorporating several attitude pointing modes
    runDvBurn(burnTime, -1, velAsteroidGuidance.attRefOutMsg)
    runSensorSciencePointing(P0)
    runDvBurn(burnTime, -1, velAsteroidGuidance.attRefOutMsg)

    # Get access to dynManager translational states for future access to the states
    velRef = scObject.dynManager.getStateObject(scObject.hub.nameOfHubVelocity)

    # Retrieve the latest relative position and velocity components
    rN = scRec.r_BN_N[-1] - astRec.PositionVector[-1]
    vN = scRec.v_BN_N[-1] - astRec.VelocityVector[-1]

    # Conduct the first burn of a Hohmann transfer from r0 to r1
    rData = np.linalg.norm(rN)
    vData = np.linalg.norm(vN)
    at = (rData + r1) * .5
    v0p = np.sqrt((2 * mu / rData) - (mu / at))
    vHat = vN / vData
    vVt = vN + vHat * (v0p - vData)
    # Update state manager's velocity
    velRef.setState(vVt)

    # Run thruster burn mode along with sun-pointing during the transfer orbit
    runDvBurn(burnTime, -1, velAsteroidGuidance.attRefOutMsg)

    # Retrieve the latest relative position and velocity components
    rN = scRec.r_BN_N[-1] - astRec.PositionVector[-1]
    vN = scRec.v_BN_N[-1] - astRec.VelocityVector[-1]

    # Conduct the second burn of the Hohmann transfer to arrive in a circular orbit at r1
    rData = np.linalg.norm(rN)
    vData = np.linalg.norm(vN)
    v1p = np.sqrt(mu / rData)
    vHat = vN / vData
    vVt2 = vN + vHat * (v1p - vData)
    # Update state manager's velocity
    velRef.setState(vVt2)

    # Run thruster burn visualization along with attitude pointing modes
    runDvBurn(burnTime, -1, velAsteroidGuidance.attRefOutMsg)
    runSensorSciencePointing(P1)
    runDvBurn(burnTime, -1, velAsteroidGuidance.attRefOutMsg)

    # Retrieve the latest relative position and velocity components
    rN = scRec.r_BN_N[-1] - astRec.PositionVector[-1]
    vN = scRec.v_BN_N[-1] - astRec.VelocityVector[-1]

    posData1 = scRec.r_BN_N  # inertial pos. wrt. Sun
    posData2 = scRec.r_BN_N - astRec.PositionVector  # relative pos. wrt. Asteroid

    # Call plotting function: plotOrbits
    figureList = plotOrbits(scRec.times(), posData1, posData2, rP, diam)

    if show_plots:
        plt.show()

    # Close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    # Unload Spice kernels
    gravFactory.unloadSpiceKernels()

    return figureList


def plotOrbits(timeAxis, posData1, posData2, rP, diam):
    fileName = os.path.basename(os.path.splitext(__file__)[0])

    plt.close("all")  # Clears out plots from earlier test runs
    figureList = {}

    # Create 3D plot for arrival to Asteroid
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the asteroid as a sphere
    planetRadius = 0.5 * diam  # m
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = planetRadius * np.cos(u) * np.sin(v)
    y = planetRadius * np.sin(u) * np.sin(v)
    z = planetRadius * np.cos(v)
    ax.plot_surface(x, y, z, color='green', alpha=0.3)
    
    # Plot the spacecraft trajectory
    ax.plot(posData2[:, 0], posData2[:, 1], posData2[:, 2], 
            color='orangered', linewidth=2, label='Simulated Flight')
    
    # Draw desired circular capture orbit
    theta = np.linspace(0, 2 * np.pi, 100)
    x_orbit = rP * np.cos(theta)
    y_orbit = rP * np.sin(theta)
    z_orbit = np.zeros_like(theta)
    ax.plot(x_orbit, y_orbit, z_orbit, '--', color='#555555', 
            linewidth=1.5, label='Desired Circ. Capture Orbit')
    
    # Set labels and title
    ax.set_xlabel('X Distance [m]')
    ax.set_ylabel('Y Distance [m]')
    ax.set_zlabel('Z Distance [m]')
    ax.set_title('Spacecraft Orbit around Asteroid')
    
    # Equal aspect ratio for all axes
    max_range = np.array([
        np.max(posData2[:, 0]) - np.min(posData2[:, 0]),
        np.max(posData2[:, 1]) - np.min(posData2[:, 1]),
        np.max(posData2[:, 2]) - np.min(posData2[:, 2])
    ]).max() / 2.0
    
    mid_x = (np.max(posData2[:, 0]) + np.min(posData2[:, 0])) / 2
    mid_y = (np.max(posData2[:, 1]) + np.min(posData2[:, 1])) / 2
    mid_z = (np.max(posData2[:, 2]) + np.min(posData2[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    ax.grid(True)
    
    # Set initial view angle
    ax.view_init(elev=30, azim=45)
    
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)

    return figureList


if __name__ == "__main__":
    run(
        True  # show_plots
    )

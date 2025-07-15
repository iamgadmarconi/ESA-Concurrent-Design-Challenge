import math
import os

import matplotlib.pyplot as plt
import numpy as np
from Basilisk.architecture import astroConstants, messaging, sysModel
from Basilisk.fswAlgorithms import (attTrackingError, locationPointing,
                                    mrpFeedback, rwMotorTorque,
                                    smallBodyWaypointFeedback)
from Basilisk.simulation import (ephemerisConverter, extForceTorque,
                                 planetEphemeris, planetNav, radiationPressure,
                                 reactionWheelStateEffector, simpleNav,
                                 spacecraft)
from Basilisk.utilities import (SimulationBaseClass, macros, orbitalMotion,
                                simIncludeGravBody, simIncludeRW,
                                unitTestSupport, vizSupport)

try:
    from Basilisk.simulation import vizInterface
    vizFound = True
except ImportError:
    vizFound = False

from .plots import (plot_control, plot_position, plot_sc_att, plot_sc_rate,
                    plot_velocity)

fileName = os.path.basename(os.path.splitext(__file__)[0])



class TouchAndGoWaypoint(sysModel.SysModel):
    """
    Basilisk custom simulation module that generates a time-varying
    Touch-and-Go (TAG) waypoint for a spacecraft operating in the vicinity
    of a small body (e.g., comet or asteroid).

    The module produces position and velocity set-points that guide a
    small-body feedback controller (``smallBodyWaypointFeedback``) through
    the four phases of a TAG mission:

    1. **ORBIT** - maintain a synchronously rotating circular orbit at
       ``initialRadius``.
    2. **APPROACH** - descend linearly to ``finalAltitude`` beginning at
       ``approachStartTime`` and lasting ``approachTime`` seconds.
    3. **HOVER** - station-keep at ``finalAltitude`` for ``hoverTime`` seconds.
    4. **DEPARTURE** - ascend linearly back to ``initialRadius`` over
       ``departureTime`` seconds.

    At every Basilisk simulation step the module updates

    * ``waypointModule.x1_ref`` - desired inertial position `[m]`
    * ``waypointModule.x2_ref`` - desired inertial velocity `[m/s]`

    allowing the downstream guidance/controller to drive the spacecraft
    accordingly.

    Parameters
    ----------
    initialRadius : float
        Radius of the initial circular orbit `[m]`.
    finalAltitude : float
        Target altitude above the small-body centre during hover `[m]`.
    approachTime : float
        Duration of the descent phase `[s]`.
    hoverTime : float
        Duration of the hover/station-keeping phase `[s]`.
    departureTime : float
        Duration of the ascent phase `[s]`.
    orbitPeriod : float
        Period of the synchronous orbit used to match the body's rotation `[s]`.
    approachStartTime : float
        Simulation time at which the approach phase should begin `[s]`.
    orbitPlane : str
        Plane in which the circular orbit is defined. One of ``'xy'``,
        ``'yz'`` or ``'xz'``.
    startPosition : list[float]
        Initial waypoint position expressed in the inertial frame `[m]`.
    startVelocity : list[float]
        Initial waypoint velocity expressed in the inertial frame `[m/s]`.

    Notes
    -----
    • The class inherits from :class:`Basilisk.architecture.sysModel.SysModel`
      and therefore implements the standard ``Reset`` and ``UpdateState``
      interface methods expected by the Basilisk simulation engine.
    • Internally the module keeps track of the current ``missionPhase`` and
      automatically transitions between phases based on the configured
      timings.
    • Controller gains can be adapted externally by subscribing to the
      ``missionPhase`` attribute, as demonstrated in ``run()`` using the
      auxiliary ``GainUpdater`` class.

    Algorithm
    ---------
    The waypoint is generated analytically at every call to ``UpdateState``:

    1. Convert the current simulation time from nanoseconds to seconds.
    2. Update the finite-state machine and switch between **ORBIT**, **APPROACH**,
       **HOVER** and **DEPARTURE** when the configured timing thresholds are
       crossed.
    3. Compute the body-fixed angular rate ``omega = 2 * pi / orbitPeriod`` and
       the instantaneous rotation angle ``angle = omega * t``.
    4. Determine the desired radial distance ``radius``:
       - ORBIT: constant ``initialRadius``.
       - APPROACH: linear interpolation from ``initialRadius`` to
         ``finalAltitude`` over ``approachTime``.
       - HOVER: constant ``finalAltitude``.
       - DEPARTURE: linear interpolation back to ``initialRadius`` over
         ``departureTime``.
    5. Map the polar coordinates ``(radius, angle)`` into Cartesian
       coordinates in the selected ``orbitPlane`` (``'xy'``, ``'yz'`` or
       ``'xz'``).
    6. Build the velocity set-point by combining the tangential component
       ``radius * omega`` with an explicit radial term when the radius is
       changing (APPROACH/DEPARTURE). This yields a continuously
       differentiable trajectory.
    7. Write the resulting position and velocity vectors to
       ``waypointModule.x1_ref`` and ``waypointModule.x2_ref`` so that the
       downstream ``smallBodyWaypointFeedback`` controller can track them.

    This closed-form, time-parameterised solution avoids numerical
    integration, making the reference trajectory inexpensive to compute yet
    perfectly repeatable.
    """

    def __init__(self):
        super(TouchAndGoWaypoint, self).__init__()
        
        # Initialize configuration parameters
        self.initialRadius = 2000.0         # Initial orbit radius (meters)
        self.finalAltitude = 100.0          # Final approach altitude (meters)
        self.approachTime = 3600.0          # Time to approach (seconds)
        self.hoverTime = 600.0              # Time to hover at lowest altitude (seconds)
        self.departureTime = 3600.0         # Time to depart (seconds)
        self.orbitPeriod = 12.296057 * 3600.0  # Orbital period to match asteroid rotation (seconds)
        self.waypointModule = None          # Link to the smallBodyWaypointFeedback module
        self.startPosition = [2000., 0., 0.] # Initial waypoint position
        self.startVelocity = [0.0, 0.0, 0.0] # Initial waypoint velocity
        self.orbitPlane = 'xy'              # Orbit plane (xy, yz, or xz)
        self.approachStartTime = 1800.0     # When to start the approach phase (seconds)
        
        # Internal state variables
        self.missionPhase = "ORBIT"         # ORBIT, APPROACH, HOVER, DEPARTURE
        self.phaseStartTime = 0.0           # When the current phase started
        self.prevTime = 0.0                 # Previous update time for improved logging

    def Reset(self, CurrentSimNanos):
        """
        Reset method initializes the waypoint
        """
        # Make sure the waypoint module is assigned
        if self.waypointModule is None:
            self.bskLogger.bskLog(
                sysModel.BSK_ERROR, 
                "TouchAndGoWaypoint: No waypoint module assigned!"
            )
            return
            
        # Set initial waypoint position and velocity
        self.waypointModule.x1_ref = self.startPosition
        self.waypointModule.x2_ref = self.startVelocity
        
        # Reset mission phase
        self.missionPhase = "ORBIT"
        self.phaseStartTime = 0.0
        self.prevTime = 0.0
        
        self.bskLogger.bskLog(
            sysModel.BSK_INFORMATION, 
            f"TouchAndGoWaypoint: Reset with initial position {self.startPosition}"
        )
        
    def UpdateState(self, CurrentSimNanos):
        """
        UpdateState method computes the waypoint position and velocity based on the current time
        and mission phase
        """
        # Check if waypoint module is assigned
        if self.waypointModule is None:
            return
            
        # Convert time to seconds
        currentTime = CurrentSimNanos * macros.NANO2SEC
        
        # Handle mission phase transitions
        if self.missionPhase == "ORBIT" and currentTime >= self.approachStartTime:
            self.missionPhase = "APPROACH"
            self.phaseStartTime = currentTime
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION,
                f"TouchAndGoWaypoint: Starting APPROACH phase at t={currentTime:.1f}s"
            )
        elif self.missionPhase == "APPROACH" and currentTime >= (self.phaseStartTime + self.approachTime):
            self.missionPhase = "HOVER"
            self.phaseStartTime = currentTime
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION,
                f"TouchAndGoWaypoint: Starting HOVER phase at t={currentTime:.1f}s, target altitude={self.finalAltitude:.1f}m"
            )
        elif self.missionPhase == "HOVER" and currentTime >= (self.phaseStartTime + self.hoverTime):
            self.missionPhase = "DEPARTURE"
            self.phaseStartTime = currentTime
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION,
                f"TouchAndGoWaypoint: Starting DEPARTURE phase at t={currentTime:.1f}s"
            )
        
        # More frequent status logging during hover
        if self.missionPhase == "HOVER" and (currentTime - self.prevTime) >= 60.0:  # Log every minute during hover
            timeInPhase = currentTime - self.phaseStartTime
            timeLeft = self.hoverTime - timeInPhase
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION,
                f"TouchAndGoWaypoint: HOVERING at t={currentTime:.1f}s, {timeInPhase:.1f}s into hover, {timeLeft:.1f}s remaining"
            )
            self.prevTime = currentTime
            
        # Calculate angular velocity based on orbit period (to match asteroid rotation)
        angular_velocity = 2.0 * math.pi / self.orbitPeriod
        
        # Calculate base position for the current rotational state
        angle = angular_velocity * currentTime
        
        # Calculate radius based on mission phase
        radius = self.initialRadius
        
        if self.missionPhase == "APPROACH":
            # Linear descent from initial radius to final altitude
            phase_progress = (currentTime - self.phaseStartTime) / self.approachTime
            # Clamp to ensure we don't overshoot due to timing issues
            phase_progress = min(phase_progress, 1.0)  
            radius = self.initialRadius - phase_progress * (self.initialRadius - self.finalAltitude)
            
        elif self.missionPhase == "HOVER":
            # Hold at final altitude - explicitly set to ensure precision
            radius = self.finalAltitude
            
        elif self.missionPhase == "DEPARTURE":
            # Linear ascent from final altitude back to initial radius
            phase_progress = (currentTime - self.phaseStartTime) / self.departureTime
            phase_progress = min(phase_progress, 1.0)  # Cap at 1.0
            radius = self.finalAltitude + phase_progress * (self.initialRadius - self.finalAltitude)
        
        # Calculate new position and velocity based on orbit plane
        if self.orbitPlane == 'xy':
            # Circular orbit in XY plane
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.0
            
            # Calculate velocity (derivative of position)
            # For constant radius: v = r * omega * [-sin(omega * t), cos(omega * t), 0]
            # For changing radius: need to add radial velocity component
            radial_velocity = 0.0
            if self.missionPhase == "APPROACH":
                radial_velocity = -(self.initialRadius - self.finalAltitude) / self.approachTime
            elif self.missionPhase == "HOVER":
                # Explicitly ensure zero radial velocity during hover
                radial_velocity = 0.0
            elif self.missionPhase == "DEPARTURE":
                radial_velocity = (self.initialRadius - self.finalAltitude) / self.departureTime
                
            vx = -radius * angular_velocity * math.sin(angle) + radial_velocity * math.cos(angle)
            vy = radius * angular_velocity * math.cos(angle) + radial_velocity * math.sin(angle)
            vz = 0.0
            
        elif self.orbitPlane == 'yz':
            # Circular orbit in YZ plane
            x = 0.0
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            radial_velocity = 0.0
            if self.missionPhase == "APPROACH":
                radial_velocity = -(self.initialRadius - self.finalAltitude) / self.approachTime
            elif self.missionPhase == "HOVER":
                # Explicitly ensure zero radial velocity during hover
                radial_velocity = 0.0
            elif self.missionPhase == "DEPARTURE":
                radial_velocity = (self.initialRadius - self.finalAltitude) / self.departureTime
                
            vx = 0.0
            vy = -radius * angular_velocity * math.sin(angle) + radial_velocity * math.cos(angle)
            vz = radius * angular_velocity * math.cos(angle) + radial_velocity * math.sin(angle)
            
        elif self.orbitPlane == 'xz':
            # Circular orbit in XZ plane
            x = radius * math.cos(angle)
            y = 0.0
            z = radius * math.sin(angle)
            
            radial_velocity = 0.0
            if self.missionPhase == "APPROACH":
                radial_velocity = -(self.initialRadius - self.finalAltitude) / self.approachTime
            elif self.missionPhase == "HOVER":
                # Explicitly ensure zero radial velocity during hover
                radial_velocity = 0.0
            elif self.missionPhase == "DEPARTURE":
                radial_velocity = (self.initialRadius - self.finalAltitude) / self.departureTime
                
            vx = -radius * angular_velocity * math.sin(angle) + radial_velocity * math.cos(angle)
            vy = 0.0
            vz = radius * angular_velocity * math.cos(angle) + radial_velocity * math.sin(angle)
            
        else:
            # Default to XY plane if invalid orbit plane specified
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0.0
            
            radial_velocity = 0.0
            if self.missionPhase == "APPROACH":
                radial_velocity = -(self.initialRadius - self.finalAltitude) / self.approachTime
            elif self.missionPhase == "HOVER":
                # Explicitly ensure zero radial velocity during hover
                radial_velocity = 0.0
            elif self.missionPhase == "DEPARTURE":
                radial_velocity = (self.initialRadius - self.finalAltitude) / self.departureTime
                
            vx = -radius * angular_velocity * math.sin(angle) + radial_velocity * math.cos(angle)
            vy = radius * angular_velocity * math.cos(angle) + radial_velocity * math.sin(angle)
            vz = 0.0
            
        # Update the waypoint module
        self.waypointModule.x1_ref = [x, y, z]
        self.waypointModule.x2_ref = [vx, vy, vz]
        
        # Log waypoint updates periodically
        if int(currentTime) % 600 == 0:  # Every 5 minutes
            self.bskLogger.bskLog(
                sysModel.BSK_INFORMATION, 
                f"TouchAndGoWaypoint: t={currentTime:.1f}s, phase={self.missionPhase}, radius={radius:.2f}m, position=[{x:.2f}, {y:.2f}, {z:.2f}]"
            )

def run(show_plots=True):
    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    # Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTime = macros.sec2nano(3600.0 * 24.0)  # 1 day simulation time

    # Create the simulation process
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # Create the dynamics task and specify the simulation time step information
    simulationTimeStep = macros.sec2nano(1.0)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Setup celestial object ephemeris module
    gravBodyEphem = planetEphemeris.PlanetEphemeris()
    gravBodyEphem.ModelTag = 'planetEphemeris'
    gravBodyEphem.setPlanetNames(planetEphemeris.StringVector(["320P/McNaught"]))

    oeAsteroid = planetEphemeris.ClassicElements()
    oeAsteroid.a = 3.103 * astroConstants.AU * 1000  # meters
    oeAsteroid.e = 0.6824
    oeAsteroid.i = 4.89 * macros.D2R
    oeAsteroid.Omega = 295.96 * macros.D2R
    oeAsteroid.omega = 0.67 * macros.D2R
    oeAsteroid.f = 0.0 * macros.D2R
    r_ON_N, v_ON_N = orbitalMotion.elem2rv(astroConstants.MU_SUN*(1000.**3), oeAsteroid)

    # specify celestial object orbit
    gravBodyEphem.planetElements = planetEphemeris.classicElementVector([oeAsteroid])
    gravBodyEphem.rightAscension = planetEphemeris.DoubleVector([0. * macros.D2R])
    gravBodyEphem.declination = planetEphemeris.DoubleVector([90. * macros.D2R])
    gravBodyEphem.lst0 = planetEphemeris.DoubleVector([0.0 * macros.D2R])
    gravBodyEphem.rotRate = planetEphemeris.DoubleVector([360 * macros.D2R / (12.296057 * 3600.)])  # rad/sec

    # setup Sun Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravFactory.createSun()

    # Create a sun spice message, zero it out, required by srp
    sunPlanetStateMsgData = messaging.SpicePlanetStateMsgPayload()
    sunPlanetStateMsg = messaging.SpicePlanetStateMsg()
    sunPlanetStateMsg.write(sunPlanetStateMsgData)

    # Create a sun ephemeris message, zero it out, required by nav filter
    sunEphemerisMsgData = messaging.EphemerisMsgPayload()
    sunEphemerisMsg = messaging.EphemerisMsg()
    sunEphemerisMsg.write(sunEphemerisMsgData)

    mu = 4.892  # m^3/s^2
    asteroid = gravFactory.createCustomGravObject("320P/McNaught", mu, 
                                                modelDictionaryKey="asteroid2"
                                                )
    asteroid.planetBodyInMsg.subscribeTo(gravBodyEphem.planetOutMsgs[0])

    # create SC object
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "bskSat"
    gravFactory.addBodiesTo(scObject)

    # Create the position and velocity of states of the s/c wrt the small body hill frame origin
    r_BO_N = np.array([2000., 0., 0.]) # Position of the spacecraft relative to the body
    v_BO_N = np.array([0., 0., 0.])  # Velocity of the spacecraft relative to the body

    # Create the inertial position and velocity of the s/c
    r_BN_N = np.add(r_BO_N, r_ON_N)
    v_BN_N = np.add(v_BO_N, v_ON_N)

    # Set the truth ICs for the spacecraft position and velocity
    scObject.hub.r_CN_NInit = r_BN_N  # m   - r_BN_N
    scObject.hub.v_CN_NInit = v_BN_N  # m/s - v_BN_N

    I = [82.12, 0.0, 0.0, 0.0, 98.40, 0.0, 0.0, 0.0, 121.0]

    mass = 330.  # kg
    scObject.hub.mHub = mass
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

    # Set the truth ICs for the spacecraft attitude and rate
    scObject.hub.sigma_BNInit = np.array([0.3, 0.7, 0.2])  # rad
    scObject.hub.omega_BN_BInit = np.array([0.0, 0.0, 0.0])  # rad/s

    # Create RWs
    rwFactory = simIncludeRW.rwFactory()

    varRWModel = messaging.BalancedWheels
    c = 2**(-0.5)

    RW1 = rwFactory.create('Honeywell_HR16', [c, 0, c], maxMomentum=50., RWModel=varRWModel, u_max=0.2)
    RW2 = rwFactory.create('Honeywell_HR16', [0, c, c], maxMomentum=50., RWModel=varRWModel, u_max=0.2)
    RW3 = rwFactory.create('Honeywell_HR16', [-c, 0, c], maxMomentum=50., RWModel=varRWModel, u_max=0.2)
    RW4 = rwFactory.create('Honeywell_HR16', [0, -c, c], maxMomentum=50., RWModel=varRWModel, u_max=0.2)

    # create RW object container and tie to spacecraft object
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    rwConfigMsg = rwFactory.getConfigMessage()

    # Create an SRP model
    srp = radiationPressure.RadiationPressure()  # default model is the SRP_CANNONBALL_MODEL
    srp.area = 1.  # m^3
    srp.coefficientReflection = 1.9
    scObject.addDynamicEffector(srp)
    srp.sunEphmInMsg.subscribeTo(sunPlanetStateMsg)

    # Create an ephemeris converter
    ephemConverter = ephemerisConverter.EphemerisConverter()
    ephemConverter.ModelTag = "ephemConverter"
    ephemConverter.addSpiceInputMsg(gravBodyEphem.planetOutMsgs[0])

    # Set up simpleNav for s/c "measurements"
    simpleNavMeas = simpleNav.SimpleNav()
    simpleNavMeas.ModelTag = 'SimpleNav'
    simpleNavMeas.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    pos_sigma_sc = 30.0       # Position accuracy (m) - less critical for pure ADCS
    vel_sigma_sc = 0.     # Velocity accuracy (m/s)
    att_sigma_sc = 0.0   # Attitude knowledge accuracy (rad) - ~0.0023° from star tracker spec
    rate_sigma_sc = 0.0  # Angular rate knowledge accuracy (rad/s) - from Astrix-120's 5e-4°/hr
    sun_sigma_sc = 0.0003    # Sun vector knowledge accuracy - from fine sun sensor 0.01-0.05° spec
    dv_sigma_sc = 0.00005     # Delta-v accuracy (m/s)
    p_matrix_sc = [[pos_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., pos_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., pos_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., vel_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., vel_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., vel_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., att_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., att_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., att_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_sc, 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_sc, 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_sc, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., sun_sigma_sc, 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., sun_sigma_sc, 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., sun_sigma_sc, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., dv_sigma_sc, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., dv_sigma_sc, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., dv_sigma_sc]]
    walk_bounds_sc = [[10.], [10.], [10.], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.], [0.], [0.], [0.], [0.], [0.]]
    simpleNavMeas.PMatrix = p_matrix_sc
    simpleNavMeas.walkBounds = walk_bounds_sc


    # Set up planetNav for comet "measurements"
    planetNavMeas = planetNav.PlanetNav()
    planetNavMeas.ephemerisInMsg.subscribeTo(ephemConverter.ephemOutMsgs[0])
    pos_sigma_p = 0.0
    vel_sigma_p = 0.0
    att_sigma_p = 0.1 * math.pi / 180.0
    rate_sigma_p = 0.3 * math.pi / 180.0
    p_matrix_p = [[pos_sigma_p, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., pos_sigma_p, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., pos_sigma_p, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., vel_sigma_p, 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., vel_sigma_p, 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., vel_sigma_p, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., att_sigma_p, 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., att_sigma_p, 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., att_sigma_p, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_p, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_p, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rate_sigma_p]]
    walk_bounds_p = [[0.], [0.], [0.], [0.], [0.], [0.], [0.005], [0.005], [0.005], [0.002], [0.002], [0.002]]
    planetNavMeas.PMatrix = p_matrix_p
    planetNavMeas.walkBounds = walk_bounds_p

    # Set up sensor science-pointing guidance module
    cameraLocation = [0.0, 0.0, -1.0]
    sciencePointGuidance = locationPointing.locationPointing()
    sciencePointGuidance.ModelTag = "sciencePointAsteroid"
    sciencePointGuidance.celBodyInMsg.subscribeTo(ephemConverter.ephemOutMsgs[0])
    sciencePointGuidance.scTransInMsg.subscribeTo(simpleNavMeas.transOutMsg)
    sciencePointGuidance.scAttInMsg.subscribeTo(simpleNavMeas.attOutMsg)
    sciencePointGuidance.pHat_B = cameraLocation  # y-axis set for science-pointing sensor
    sciencePointGuidance.useBoresightRateDamping = 1
    scSim.AddModelToTask(simTaskName, sciencePointGuidance)

    # Attitude error configuration
    trackingError = attTrackingError.attTrackingError()
    trackingError.ModelTag = "trackingError"
    trackingError.attRefInMsg.subscribeTo(sciencePointGuidance.attRefOutMsg)

    # Specify the vehicle configuration message
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = I
    vehicleConfigOut.CoM_B = [0.0, 0.0, 0.0]
    vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # Attitude controller configuration
    mrpFeedbackControl = mrpFeedback.mrpFeedback()
    mrpFeedbackControl.ModelTag = "mrpFeedbackControl"
    mrpFeedbackControl.guidInMsg.subscribeTo(trackingError.attGuidOutMsg)
    mrpFeedbackControl.vehConfigInMsg.subscribeTo(vcConfigMsg)
    mrpFeedbackControl.K = 10.0
    mrpFeedbackControl.Ki = 0.001
    mrpFeedbackControl.P = 20.
    mrpFeedbackControl.integralLimit = 2. / mrpFeedbackControl.Ki * 0.1

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(rwConfigMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpFeedbackControl.cmdTorqueOutMsg)
    rwMotorTorqueObj.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)

    # Connect the navigation output messages
    trackingError.attNavInMsg.subscribeTo(simpleNavMeas.attOutMsg)

    # Create the Lyapunov feedback controller
    waypointFeedback = smallBodyWaypointFeedback.SmallBodyWaypointFeedback()
    waypointFeedback.asteroidEphemerisInMsg.subscribeTo(planetNavMeas.ephemerisOutMsg)
    waypointFeedback.sunEphemerisInMsg.subscribeTo(sunEphemerisMsg)
    waypointFeedback.navAttInMsg.subscribeTo(simpleNavMeas.attOutMsg)
    waypointFeedback.navTransInMsg.subscribeTo(simpleNavMeas.transOutMsg)
    waypointFeedback.A_sc = 1.  # Surface area of the spacecraft, m^2
    waypointFeedback.M_sc = mass  # Mass of the spacecraft, kg
    waypointFeedback.IHubPntC_B = unitTestSupport.np2EigenMatrix3d(I)  # sc inertia
    waypointFeedback.mu_ast = mu  # Gravitational constant of the asteroid
    
    # Initial waypoint values
    waypointFeedback.x1_ref = [2000., 0., 0.]
    waypointFeedback.x2_ref = [0.0, 0.0, 0.0]

    # Create the touch-and-go waypoint module
    touchAndGoWaypoint = TouchAndGoWaypoint()
    touchAndGoWaypoint.ModelTag = "touchAndGoWaypoint"
    
    # Configure the touch-and-go parameters
    touchAndGoWaypoint.initialRadius = 2000.0     # Initial orbit altitude (meters)
    touchAndGoWaypoint.finalAltitude = 1120.0     # Final approach altitude (meters)
    touchAndGoWaypoint.approachTime = 4*3600.0    # 4 hours approach
    touchAndGoWaypoint.hoverTime = 2*3600.0       # 2 hours hovering
    touchAndGoWaypoint.departureTime = 4*3600.0   # 4 hours departure
    
    # Configure when to start the approach phase (from the beginning of the simulation)
    touchAndGoWaypoint.approachStartTime = 12*3600.0  # Start approach after 12 hours orbit
    
    # Match the asteroid's rotation period
    asteroidRotationPeriod = 12.296057 * 3600.0   # Convert hours to seconds
    touchAndGoWaypoint.orbitPeriod = asteroidRotationPeriod
    
    touchAndGoWaypoint.startPosition = [2000., 0., 0.]  # Initial position
    touchAndGoWaypoint.startVelocity = [0.0, 0.0, 0.0]  # Initial velocity
    touchAndGoWaypoint.orbitPlane = 'xy'                # Orbit plane
    
    # Link the waypoint module to the updater
    touchAndGoWaypoint.waypointModule = waypointFeedback

    # Set up external force/torque module
    extForceTorqueModule = extForceTorque.ExtForceTorque()
    extForceTorqueModule.cmdForceBodyInMsg.subscribeTo(waypointFeedback.forceOutMsg)
    scObject.addDynamicEffector(extForceTorqueModule)

    # Add models to the simulation task
    scSim.AddModelToTask(simTaskName, scObject, 200)
    scSim.AddModelToTask(simTaskName, srp, 199)
    scSim.AddModelToTask(simTaskName, gravBodyEphem, 198)
    scSim.AddModelToTask(simTaskName, rwStateEffector, 197)
    scSim.AddModelToTask(simTaskName, ephemConverter, 197)
    scSim.AddModelToTask(simTaskName, simpleNavMeas, 196)
    scSim.AddModelToTask(simTaskName, planetNavMeas, 195)
    scSim.AddModelToTask(simTaskName, trackingError, 106)
    scSim.AddModelToTask(simTaskName, mrpFeedbackControl, 105)
    scSim.AddModelToTask(simTaskName, extForceTorqueModule, 82)
    scSim.AddModelToTask(simTaskName, rwMotorTorqueObj, 81)
    scSim.AddModelToTask(simTaskName, waypointFeedback, 78)
    
    # Add the touch-and-go waypoint module to the task
    # Make sure this executes before the waypoint feedback for proper sequence
    scSim.AddModelToTask(simTaskName, touchAndGoWaypoint, 79)

    # Setup data logging
    sc_truth_recorder = scObject.scStateOutMsg.recorder()
    ast_truth_recorder = gravBodyEphem.planetOutMsgs[0].recorder()
    ast_ephemeris_recorder = ephemConverter.ephemOutMsgs[0].recorder()
    ast_ephemeris_meas_recorder = planetNavMeas.ephemerisOutMsg.recorder()
    sc_meas_recorder = simpleNavMeas.transOutMsg.recorder()
    sc_att_meas_recorder = simpleNavMeas.attOutMsg.recorder()
    requested_control_recorder = waypointFeedback.forceOutMsg.recorder()
    attitude_error_recorder = trackingError.attGuidOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, sc_truth_recorder)
    scSim.AddModelToTask(simTaskName, ast_truth_recorder)
    scSim.AddModelToTask(simTaskName, sc_meas_recorder)
    scSim.AddModelToTask(simTaskName, sc_att_meas_recorder)
    scSim.AddModelToTask(simTaskName, ast_ephemeris_recorder)
    scSim.AddModelToTask(simTaskName, ast_ephemeris_meas_recorder)
    scSim.AddModelToTask(simTaskName, requested_control_recorder)
    scSim.AddModelToTask(simTaskName, attitude_error_recorder)

    fileName = 'touchAndGo'

    if vizSupport.vizFound:
        vizInterface = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                                            , saveFile=fileName, rwEffectorList=rwStateEffector
                                                            )

        vizSupport.createStandardCamera(vizInterface, setMode=1, spacecraftName=scObject.ModelTag,
                                        fieldOfView=10 * macros.D2R,
                                        displayName="10˚ FOV Camera",
                                        pointingVector_B=[0, 0, -1], position_B=cameraLocation)

        vizInterface.settings.showCSLabels = 1
        vizInterface.settings.planetCSon = 1
        vizInterface.settings.orbitLinesOn = -1

    # initialize Simulation
    scSim.InitializeSimulation()
    
    # Log mission timeline
    print("\n=== Touch-and-Go Mission Timeline ===")
    print(f"T+00:00 - T+{touchAndGoWaypoint.approachStartTime/3600.0:05.2f} hours: Comet Synchronous Orbit at {touchAndGoWaypoint.initialRadius:.0f}m")
    approachEnd = touchAndGoWaypoint.approachStartTime + touchAndGoWaypoint.approachTime
    print(f"T+{touchAndGoWaypoint.approachStartTime/3600.0:05.2f} - T+{approachEnd/3600.0:05.2f} hours: Descent to {touchAndGoWaypoint.finalAltitude:.0f}m")
    hoverEnd = approachEnd + touchAndGoWaypoint.hoverTime
    print(f"T+{approachEnd/3600.0:05.2f} - T+{hoverEnd/3600.0:05.2f} hours: Hover at {touchAndGoWaypoint.finalAltitude:.0f}m")
    departureEnd = hoverEnd + touchAndGoWaypoint.departureTime
    print(f"T+{hoverEnd/3600.0:05.2f} - T+{departureEnd/3600.0:05.2f} hours: Ascent to {touchAndGoWaypoint.initialRadius:.0f}m")
    print(f"T+{departureEnd/3600.0:05.2f} - T+{simulationTime*macros.NANO2SEC/3600.0:05.2f} hours: Comet Synchronous Orbit at {touchAndGoWaypoint.initialRadius:.0f}m")
    print("=====================================\n")

    # Set controller gains
    # Use different gains for different mission phases
    def updateControllerGains(whichPhase):
        if whichPhase == "APPROACH" or whichPhase == "DEPARTURE":
            # Lower gains during approach/departure for smoother motion
            waypointFeedback.K1 = unitTestSupport.np2EigenMatrix3d(1 * np.array([5e-4, 0e-5, 0e-5, 
                                                                                0e-5, 5e-4, 0e-5, 
                                                                                0e-5, 0e-5, 5e-4]))

            waypointFeedback.K2 = unitTestSupport.np2EigenMatrix3d(1 * np.array([1., 0., 0.,
                                                                                0., 1., 0., 
                                                                                0., 0., 1.]))
        elif whichPhase == "HOVER":
            # Higher position gain during hover for better position holding
            waypointFeedback.K1 = unitTestSupport.np2EigenMatrix3d(2 * np.array([5e-4, 0e-5, 0e-5, 
                                                                                0e-5, 5e-4, 0e-5, 
                                                                                0e-5, 0e-5, 5e-4]))

            waypointFeedback.K2 = unitTestSupport.np2EigenMatrix3d(10 * np.array([1., 0., 0.,
                                                                                0., 1., 0., 
                                                                                0., 0., 1.]))
        else:  # Default/ORBIT
            waypointFeedback.K1 = unitTestSupport.np2EigenMatrix3d(1 * np.array([5e-4, 0e-5, 0e-5, 
                                                                                0e-5, 5e-4, 0e-5, 
                                                                                0e-5, 0e-5, 5e-4]))

            waypointFeedback.K2 = unitTestSupport.np2EigenMatrix3d(1 * np.array([1., 0., 0.,
                                                                                0., 1., 0., 
                                                                                0., 0., 1.]))
    
    # Set the initial gains
    updateControllerGains("ORBIT")
    
    # Subscribe to messages from the waypoint module to update controller gains
    # Set up a callback to change gains when mission phase changes
    class GainUpdater(sysModel.SysModel):
        def __init__(self):
            super(GainUpdater, self).__init__()
            self.ModelTag = "gainUpdater"
            self.lastPhase = "ORBIT"
        
        def Reset(self, CurrentSimNanos):
            self.lastPhase = "ORBIT"
            
        def UpdateState(self, CurrentSimNanos):
            if touchAndGoWaypoint.missionPhase != self.lastPhase:
                # Phase changed, update gains
                updateControllerGains(touchAndGoWaypoint.missionPhase)
                self.bskLogger.bskLog(
                    sysModel.BSK_INFORMATION,
                    f"GainUpdater: Updated gains for phase {touchAndGoWaypoint.missionPhase}"
                )
                self.lastPhase = touchAndGoWaypoint.missionPhase
    
    # Create the gain updater
    gainUpdater = GainUpdater()
    scSim.AddModelToTask(simTaskName, gainUpdater, 77)  # Execute right before waypointFeedback

    # configure a simulation stop time and execute the simulation run
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    # retrieve logged spacecraft position relative to asteroid
    r_BN_N_truth = sc_truth_recorder.r_BN_N
    r_BN_N_meas = sc_meas_recorder.r_BN_N
    v_BN_N_truth = sc_truth_recorder.v_BN_N
    v_BN_N_meas = sc_meas_recorder.v_BN_N
    sigma_BN_truth = sc_truth_recorder.sigma_BN
    sigma_BN_meas = sc_att_meas_recorder.sigma_BN
    omega_BN_B_truth = sc_truth_recorder.omega_BN_B
    omega_BN_B_meas = sc_att_meas_recorder.omega_BN_B
    r_AN_N = ast_truth_recorder.PositionVector
    v_AN_N = ast_truth_recorder.VelocityVector
    u_requested = requested_control_recorder.forceRequestBody

    # Compute the relative position and velocity of the s/c in the small body hill frame
    r_BO_O_truth = []
    v_BO_O_truth = []
    r_BO_O_meas = []
    v_BO_O_meas = []
    np.set_printoptions(precision=15)
    for rd_N, vd_N, rc_N, vc_N, rd_N_meas, vd_N_meas in zip(r_BN_N_truth, v_BN_N_truth, r_AN_N, v_AN_N, r_BN_N_meas, v_BN_N_meas):
        # Truth values
        r_BO_O, v_BO_O = orbitalMotion.rv2hill(rc_N, vc_N, rd_N, vd_N)
        r_BO_O_truth.append(r_BO_O)
        v_BO_O_truth.append(v_BO_O)

        # Measurement values
        r_BO_O, v_BO_O = orbitalMotion.rv2hill(rc_N, vc_N, rd_N_meas, vd_N_meas)
        r_BO_O_meas.append(r_BO_O)
        v_BO_O_meas.append(v_BO_O)

    # Calculate distance to asteroid surface
    distances = []
    times = sc_truth_recorder.times() * macros.NANO2SEC
    for i, r_BO_O in enumerate(r_BO_O_truth):
        distance = np.linalg.norm(r_BO_O)
        distances.append(distance)

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, distances)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Distance from asteroid center [m]')
    ax.set_title('Spacecraft Distance During Touch-and-Go Maneuver')
    ax.grid(True)
    
    figureList = {}
    figureList['distance'] = fig
    
    # Original plots from the example
    plot_position(times, np.array(r_BO_O_truth), np.array(r_BO_O_meas))
    figureList['position'] = plt.figure(2)

    plot_velocity(times, np.array(v_BO_O_truth), np.array(v_BO_O_meas))
    figureList['velocity'] = plt.figure(3)

    plot_control(times, np.array(u_requested))
    figureList['control'] = plt.figure(4)

    plot_sc_att(times, np.array(sigma_BN_truth), np.array(sigma_BN_meas))
    figureList['attitude'] = plt.figure(5)

    plot_sc_rate(times, np.array(omega_BN_B_truth), np.array(omega_BN_B_meas))
    figureList['rate'] = plt.figure(6)

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    return figureList


if __name__ == "__main__":
    run(True)  # show_plots

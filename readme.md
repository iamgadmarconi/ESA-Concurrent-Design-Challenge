# Touch-and-Go Small-Body Mission Simulation

This repository contains a **Basilisk** simulation that demonstrates a full _touch-and-go_ (TAG) manoeuvre about the comet **320P/McNaught**. The key piece of the implementation is the custom module
`TouchAndGoWaypoint` found in `src/algorithm.py`.

## High-level algorithm

At every simulation step the waypoint module:

1. Converts current simulation time to seconds.
2. Advances a finite-state machine through the four TAG phases.
3. Computes the synchronous angular rate `omega = 2 * pi / orbitPeriod`.
4. Sets the desired radius according to phase (constant or linearly
   interpolated).
5. Maps `(radius, angle)` into Cartesian coordinates in the chosen orbit plane
   (`xy`, `yz` or `xz`).
6. Adds the proper tangential and radial velocity components.
7. Publishes position/velocity set-points to `smallBodyWaypointFeedback`.

The result is a smooth trajectory that a downstream Lyapunov controller can
track with minimal effort.

While not altering any Basilisk internals, the generator simply refreshes the
reference message that `smallBodyWaypointFeedback` already consumes, letting us
reuse its proven control logic with almost no extra complexity.

## Results

Satellite approaching and matching the velocity of the comet.
![Satellite approaching and matching the velocity of the comet](results/approach-hover-departure.gif)

Satellite performing a TAG manoeuvre.
![Satellite performing a TAG manoeuvre](results/match-velocity.gif)

## Customising the scenario

Edit the parameters at the bottom of `algorithm.py`, for example:

```python
touchAndGoWaypoint.initialRadius = 2500.0   # [m]
touchAndGoWaypoint.finalAltitude = 500.0    # [m]
...
```

## Prerequisites

* **Python 3.9+** (tested on Windows 11)
* [Basilisk](https://github.com/AGI/Basilisk) (install via the official docs)
* `numpy`, `matplotlib`

import matplotlib.pyplot as plt


def plot_position(time, r_BO_O_truth, r_BO_O_meas):
    """Plot the relative position result."""
    fig, ax = plt.subplots(3, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time, r_BO_O_meas[:, 0], 'k*', label='measurement', markersize=1)
    ax[1].plot(time, r_BO_O_meas[:, 1], 'k*', markersize=1)
    ax[2].plot(time, r_BO_O_meas[:, 2], 'k*', markersize=1)

    ax[0].plot(time, r_BO_O_truth[:, 0], label='${}^Or_{BO_{1}}$')
    ax[1].plot(time, r_BO_O_truth[:, 1], label='${}^Or_{BO_{2}}$')
    ax[2].plot(time, r_BO_O_truth[:, 2], label='${}^Or_{BO_{3}}$')

    plt.xlabel('Time [sec]')
    plt.title('Relative Spacecraft Position')

    ax[0].set_ylabel('${}^Or_{BO_1}$ [m]')
    ax[1].set_ylabel('${}^Or_{BO_2}$ [m]')
    ax[2].set_ylabel('${}^Or_{BO_3}$ [m]')

    ax[0].legend()

    return


def plot_velocity(time, v_BO_O_truth, v_BO_O_meas):
    """Plot the relative velocity result."""
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time, v_BO_O_meas[:, 0], 'k*', label='measurement', markersize=1)
    ax[1].plot(time, v_BO_O_meas[:, 1], 'k*', markersize=1)
    ax[2].plot(time, v_BO_O_meas[:, 2], 'k*', markersize=1)

    ax[0].plot(time, v_BO_O_truth[:, 0], label='truth')
    ax[1].plot(time, v_BO_O_truth[:, 1])
    ax[2].plot(time, v_BO_O_truth[:, 2])

    plt.xlabel('Time [sec]')
    plt.title('Relative Spacecraft Velocity')

    ax[0].set_ylabel('${}^Ov_{BO_1}$ [m/s]')
    ax[1].set_ylabel('${}^Ov_{BO_2}$ [m/s]')
    ax[2].set_ylabel('${}^Ov_{BO_3}$ [m/s]')

    ax[0].legend()

    return


def plot_sc_att(time, sigma_BN_truth, sigma_BN_meas):
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time, sigma_BN_meas[:, 0], 'k*', label='measurement', markersize=1)
    ax[1].plot(time, sigma_BN_meas[:, 1], 'k*', markersize=1)
    ax[2].plot(time, sigma_BN_meas[:, 2], 'k*', markersize=1)

    ax[0].plot(time, sigma_BN_truth[:, 0], label='truth')
    ax[1].plot(time, sigma_BN_truth[:, 1])
    ax[2].plot(time, sigma_BN_truth[:, 2])

    plt.xlabel('Time [sec]')

    ax[0].set_ylabel(r'$\sigma_{BN_1}$ [rad]')
    ax[1].set_ylabel(r'$\sigma_{BN_2}$ [rad]')
    ax[2].set_ylabel(r'$\sigma_{BN_3}$ [rad]')

    ax[0].legend()

    return


def plot_sc_rate(time, omega_BN_B_truth, omega_BN_B_meas):
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time, omega_BN_B_meas[:, 0], 'k*', label='measurement', markersize=1)
    ax[1].plot(time, omega_BN_B_meas[:, 1], 'k*', markersize=1)
    ax[2].plot(time, omega_BN_B_meas[:, 2], 'k*', markersize=1)

    ax[0].plot(time, omega_BN_B_truth[:, 0], label='truth')
    ax[1].plot(time, omega_BN_B_truth[:, 1])
    ax[2].plot(time, omega_BN_B_truth[:, 2])

    plt.xlabel('Time [sec]')

    ax[0].set_ylabel(r'${}^B\omega_{BN_{1}}$ [rad/s]')
    ax[1].set_ylabel(r'${}^B\omega_{BN_{2}}$ [rad/s]')
    ax[2].set_ylabel(r'${}^B\omega_{BN_{3}}$ [rad/s]')

    ax[0].legend()

    return


def plot_control(time, u):
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12,6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time, u[:, 0], 'k-', markersize=1)
    ax[1].plot(time, u[:, 1], 'k-', markersize=1)
    ax[2].plot(time, u[:, 2], 'k-', markersize=1)

    plt.xlabel('Time [sec]')

    ax[0].set_ylabel(r'$\hat{\mathbf{b}}_1$ control [N]')
    ax[1].set_ylabel(r'$\hat{\mathbf{b}}_2$ control [N]')
    ax[2].set_ylabel(r'$\hat{\mathbf{b}}_3$ control [N]')

    return

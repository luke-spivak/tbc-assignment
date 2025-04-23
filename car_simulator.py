# The class CarSimulator is a simple 2D vehicle simulator.
# The vehicle states are:
# - x: is the position on the x axis on a xy plane
# - y: is the position on the y axis on a xy plane
# - v is the vehicle speed in the direction of travel of the vehicle
# - theta: is the angle wrt the x axis (0 rad means the vehicle
#   is parallel to the x axis, in the positive direction;
#   pi/2 rad means the vehicle is parallel
#   to the y axis, in the positive direction)
# - NOTE: all units are SI: meters (m) for distances, seconds (s) for
#   time, radians (rad) for angles...
#
# (1)
# Write the method "simulatorStep", which should update
# the vehicle states, given 3 inputs:
#  - a: commanded vehicle acceleration
#  - wheel_angle: steering angle, measured at the wheels;
#    0 rad means that the wheels are "straight" wrt the vehicle.
#    A positive value means that the vehicle is turning counterclockwise
#  - dt: duration of time after which we want to provide
#    a state update (time step)
#
# (2)
# Complete the function "main". This function should run the following simulation:
# - The vehicle starts at 0 m/s
# - The vehicle drives on a straight line and accelerates from 0 m/s to 10 m/s
#   at a constant rate of 0.4 m/s^2, then it proceeds at constant speed.
# - Once reached the speed of 10 m/s, the vehicle drives in a clockwise circle of
#   roughly 100 m radius
# - the simulation ends at 100 s
#
# (3)
# - plot the vehicle's trajectory on the xy plane
# - plot the longitudinal and lateral accelerations over time
import math
import matplotlib.pyplot as plt


class CarSimulator:
    """Simulates a car's motion using a kinematic bicycle model.

    Attributes:
        wheelbase (float): Distance between front and rear wheels (m).
        x (float): X-coordinate of the car's position (m).
        y (float): Y-coordinate of the car's position (m).
        v (float): Current velocity (m/s).
        theta (float): Current heading angle (radians).
    """
    def __init__(self, wheelbase, v0, theta0):
        self.wheelbase = wheelbase
        self.x = 0
        self.y = 0 
        self.v = v0
        self.theta = theta0

    def simulatorStep(self, a, wheel_angle, dt):
        """Simulate one time step of the car's motion.

        Updates the car's velocity, heading, and position based on acceleration,
        steering angle, and time step using a kinematic bicycle model.

        Args:
            a (float): Longitudinal acceleration (m/s^2).
            wheel_angle (float): Steering angle of the front wheels (radians).
            dt (float): Duration of the time step (s).
        """
        v_old = self.v
        x_old = self.x
        y_old = self.y
        theta_old = self.theta

        self.v = v_old + a * dt

        # Calculate change in heading angle based on wheel angle and velocity
        angular_velocity = (self.v * math.tan(wheel_angle)) / self.wheelbase
        self.theta = self.theta + angular_velocity * dt

        # Update position
        avg_v = (v_old + self.v) / 2
        if abs(angular_velocity) < 1e-6:
            # Straight path
            self.x = x_old + avg_v * math.cos(theta_old) * dt
            self.y = y_old + avg_v * math.sin(theta_old) * dt
        else:
            # exact integration along a circular arc using average velocity
            self.x = x_old + (avg_v / angular_velocity) * (
                math.sin(self.theta) - math.sin(theta_old)
            )
            self.y = y_old + (avg_v / angular_velocity) * (
                math.cos(theta_old) - math.cos(self.theta)
            )


def plot_simulation(times, x_positions, y_positions, long_accels, lat_accels):
    # Plot trajectory
    plt.figure()
    plt.plot(x_positions, y_positions)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Vehicle Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # Plot accelerations
    plt.figure()
    plt.plot(times, long_accels, label="Longitudinal Acceleration")
    plt.plot(times, lat_accels, label="Lateral Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Accelerations over Time")
    plt.legend()
    plt.show()


def main():
    """Run a car simulation and plot the trajectory and accelerations.

    Simulates a car accelerating to a target speed, then turning in a circle,
    and visualizes the resulting trajectory and accelerations.
    """
    wheelbase = 4  # arbitrary 4m wheelbase
    v0 = 0
    theta0 = 0
    simulator = CarSimulator(wheelbase, v0, theta0)
    dt = 0.1  # arbitrarily set the time step to 0.1 s
    t_end = 100  # simulation ends at 100 s

    target_speed = 10  # m/s
    acceleration_rate = 0.4  # m/s^2
    turn_radius = 100.0  # m
    wheel_angle_turn = -math.atan(wheelbase / turn_radius)  # Negative for clockwise

    times = []
    x_positions = []
    y_positions = []
    long_accels = []
    lat_accels = []

    n_steps = int(t_end / dt)
    for i in range(n_steps + 1):
        if simulator.v < target_speed:
            # Accelerate in straight line until v = 10 m/s
            a = acceleration_rate
            wheel_angle = 0  # straight line
        else: 
            # Move in circle at constant velocity
            a = 0
            wheel_angle = wheel_angle_turn

        # record state and accelerations
        t = i * dt
        times.append(t)
        x_positions.append(simulator.x)
        y_positions.append(simulator.y)
        long_accels.append(a)
        # lateral accel = v^2 / R, with R = wheelbase / tan(δ)
        lat_accels.append(simulator.v**2 * math.tan(wheel_angle) / wheelbase)

        simulator.simulatorStep(a, wheel_angle, dt)
        t += dt
    
    plot_simulation(times, x_positions, y_positions, long_accels, lat_accels)

main()

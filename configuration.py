import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def draw_vector(ax, origin, vector, color='red', label=None, arrow_length_ratio=0.15, linestyle='-'):
    """Helper function to draw a 3D vector with proper arrow head."""
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm  # Normalize the vector
    
    # Scale the vector to a visible length
    scale = 0.5
    vector = vector * scale
    
    # Create the arrow
    arrow = Arrow3D([origin[0], origin[0] + vector[0]],
                    [origin[1], origin[1] + vector[1]],
                    [origin[2], origin[2] + vector[2]],
                    mutation_scale=20, lw=2, arrowstyle='-|>', color=color, linestyle=linestyle)
    ax.add_artist(arrow)
    
    # Add label if provided
    if label:
        ax.text(origin[0] + vector[0]*1.1, 
                origin[1] + vector[1]*1.1, 
                origin[2] + vector[2]*1.1, 
                label, color=color, fontsize=18)


def draw_fov_cone(ax, origin, direction, angle_deg, color='blue', alpha=0.3, resolution=20):
    """Draw a field-of-view cone for a sensor using wireframe."""
    # Normalize direction vector
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:  # Avoid division by zero
        return
    
    direction = direction / norm
    
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Create a circle in the plane perpendicular to the direction
    radius = np.sin(angle_rad) * 0.7  # Scale radius to match visualization
    height = np.cos(angle_rad) * 0.7  # Scale height to match visualization
    
    # Find perpendicular vectors to direction
    # We need to find a vector perpendicular to direction
    if abs(direction[2]) > 0.9:  # If direction is close to z-axis
        perp1 = np.array([1.0, 0.0, 0.0])  # Choose x-axis
    else:
        perp1 = np.array([0.0, 0.0, 1.0])  # Otherwise choose z-axis
    
    # Get vector perpendicular to both direction and perp1
    perp1 = perp1 - np.dot(perp1, direction) * direction
    norm1 = np.linalg.norm(perp1)
    if norm1 < 1e-10:  # Check for numerical issues
        return
    
    perp1 = perp1 / norm1
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)  # Normalize
    
    # Create circle points for the base of the cone
    theta = np.linspace(0, 2*np.pi, resolution)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    
    # Calculate points on the cone
    apex = np.array(origin)
    base_center = apex + direction * height
    
    # Draw the cone's apex to points on the base circle (lines)
    for i in range(resolution):
        point_on_circle = base_center + circle_x[i]*perp1 + circle_y[i]*perp2
        ax.plot([apex[0], point_on_circle[0]], 
                [apex[1], point_on_circle[1]], 
                [apex[2], point_on_circle[2]], 
                color=color, alpha=alpha, linestyle='-')
    
    # Draw the base circle
    base_x = []
    base_y = []
    base_z = []
    
    for i in range(resolution):
        point = base_center + circle_x[i]*perp1 + circle_y[i]*perp2
        base_x.append(point[0])
        base_y.append(point[1])
        base_z.append(point[2])
    
    # Close the circle
    base_x.append(base_x[0])
    base_y.append(base_y[0])
    base_z.append(base_z[0])
    
    ax.plot(base_x, base_y, base_z, color=color, alpha=alpha)


def draw_cuboid(ax, dimensions, color='lightgray', alpha=0.2):
    """Draw a cuboid with the given dimensions [length, width, height]."""
    # Unpack dimensions (length = x, width = y, height = z)
    length, width, height = dimensions
    
    # Define the 8 vertices of the cuboid
    x = np.array([-length/2, length/2])
    y = np.array([-width/2, width/2])
    z = np.array([-height/2, height/2])
    
    # Create arrays to store the vertices and faces
    vertices = np.array([[x_i, y_j, z_k] for x_i in x for y_j in y for z_k in z])
    
    # Define the 6 faces using indices of vertices
    faces = [
        [0, 1, 3, 2],  # -x face
        [4, 5, 7, 6],  # +x face
        [0, 2, 6, 4],  # -y face
        [1, 3, 7, 5],  # +y face
        [0, 1, 5, 4],  # -z face
        [2, 3, 7, 6]   # +z face
    ]
    
    # Plot each face as a Poly3DCollection
    for face in faces:
        face_vertices = [vertices[i] for i in face]
        poly = Poly3DCollection([face_vertices], alpha=alpha, color=color)
        ax.add_collection3d(poly)
    
    # Add the edges for better visibility
    for face in faces:
        x = [vertices[i][0] for i in face]
        y = [vertices[i][1] for i in face]
        z = [vertices[i][2] for i in face]
        x.append(x[0])  # Close the face
        y.append(y[0])
        z.append(z[0])
        ax.plot(x, y, z, color='gray', alpha=0.5, linewidth=1)


def visualize_sensor_configuration(dimensions=[1.6, 1, 1.3]):
    """
    Visualize the sensor configuration for a cuboid spacecraft.
    
    Parameters:
    -----------
    dimensions : list
        Dimensions of the spacecraft [length, width, height] in meters.
    """
    # Create figure and 3D axis with dark background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 18})
    
    # Draw the cuboid spacecraft
    draw_cuboid(ax, dimensions)
    
    # Unpack dimensions for easier reference
    length, width, height = dimensions
    
    # Draw body axes
    draw_vector(ax, [0, 0, 0], [4.5, 0, 0], color='red', label="+X", arrow_length_ratio=1)
    draw_vector(ax, [0, 0, 0], [0, 4.5, 0], color='green', label="+Y", arrow_length_ratio=1)
    draw_vector(ax, [0, 0, 0], [0, 0, 4.5], color='blue', label="+Z", arrow_length_ratio=1)
    
    # Parameters from sun_pointing.py
    star_tracker_fov = 30.0  # degrees
    sun_sensor_fov = 100.0  # degrees
    
    # Place and label sensors
    # 1. Airbus Astrix-120 IMU - near center of mass, aligned with body axes
    ax.scatter([0], [0], [0], color='white', s=100)
    ax.text(0.1, 0.1, 0.1, "IMU (2 units)", color='white', fontsize=18)
    
    # 2. Leonardo AA-STR Star Trackers - new configuration: ±30° offset from -X axis
    # Position them on the -X face
    st_angles = [-30, 30]  # Degrees offset from -X axis in the X-Y plane
    st_positions = []
    
    for i, angle in enumerate(st_angles):
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate position on the -X face
        x_pos = 0 # -length/2  # On the -X face
        y_pos = -width/2 # width/4 * np.sin(angle_rad)  # Offset based on angle
        z_pos = -height/4 if i == 0 else height/4  # Position on opposite sides of x-y plane
        
        # Direction vector (pointing outward with the specified angle)
        x_dir = 0  # -X direction
        y_dir = -1 # np.sin(angle_rad)  # Y-component based on angle
        z_dir = np.sin(angle_rad)
        
        st_positions.append(
            ([x_pos, y_pos, z_pos], 
             [x_dir, y_dir, z_dir],
             f"Star Tracker {i+1}")
        )
    
    for i, (pos, vec, label) in enumerate(st_positions):
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color='cyan', s=100)
        ax.text(pos[0]-0.1, pos[1]-0.1, pos[2], label, color='cyan', fontsize=18)
        draw_vector(ax, pos, vec, color='cyan', linestyle='--')
        draw_fov_cone(ax, pos, vec, star_tracker_fov, color='cyan', alpha=0.3)
    
    # 3. Redwire Adcole Fine Sun Sensors - new configuration for cuboid
    # 2 on +Z face at opposite corners, 2 on +X face at opposite corners
    fss_positions = []
    
    # +Z face sensors
    fss_positions.append(
        ([0, width/4, height/2], 
         [0, 0.5, 1],  # +Z direction
         "FSS 1")
    )
    
    fss_positions.append(
        ([0, -width/4, height/2], 
         [0, -0.5, 1],  # +Z direction
         "FSS 2")
    )
    
    # +X face sensors
    fss_positions.append(
        ([length/2, 0, height/4], 
         [1, 0, 0.5],  # +X direction
         "FSS 3")
    )
    
    fss_positions.append(
        ([length/2, 0, -height/4], 
         [1, 0, -0.5],  # +X direction
         "FSS 4")
    )
    
    for i, (pos, vec, label) in enumerate(fss_positions):
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color='yellow', s=80)
        ax.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, label, color='yellow', fontsize=18)
        draw_vector(ax, pos, vec, color='yellow', linestyle='--')
        draw_fov_cone(ax, pos, vec, sun_sensor_fov/2, color='yellow', alpha=0.3)
    
    # 4. Bradford Space Coarse Sun Sensors - place one on each face (6 total)
    # css_positions = [
    #     [length/2, 0, 0],   # +X face
    #     [-length/2, 0, 0],  # -X face
    #     [0, width/2, 0],    # +Y face
    #     [0, -width/2, 0],   # -Y face
    #     [0, 0, height/2],   # +Z face
    #     [0, 0, -height/2]   # -Z face
    # ]
    
    # css_vectors = [
    #     [1, 0, 0],   # +X direction
    #     [-1, 0, 0],  # -X direction
    #     [0, 1, 0],   # +Y direction
    #     [0, -1, 0],  # -Y direction
    #     [0, 0, 1],   # +Z direction
    #     [0, 0, -1]   # -Z direction
    # ]
    
    # css_labels = ["+X CSS", "-X CSS", "+Y CSS", "-Y CSS", "+Z CSS", "-Z CSS"]
    
    # for i, (pos, vec, label) in enumerate(zip(css_positions, css_vectors, css_labels)):
    #     ax.scatter([pos[0]], [pos[1]], [pos[2]], color='magenta', s=60)
    #     ax.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, label, color='magenta', fontsize=18)
    #     draw_vector(ax, pos, vec, color='magenta', linestyle='--')
    
    # Set axis limits and labels
    max_dim = max(dimensions) * 1.8
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(-max_dim, max_dim)
    
    ax.set_xlabel('X (m)', color='white', fontsize=18)
    ax.set_ylabel('Y (m)', color='white', fontsize=18)
    ax.set_zlabel('Z (m)', color='white', fontsize=18)
    ax.set_title(f'Spacecraft Sensor Configuration - Dimensions: {dimensions} m', color='white', fontsize=18)
    
    # Set tick colors to white
    ax.tick_params(colors='white', labelsize=18)
    
    # Add annotations explaining the configuration
    annotation_text = (
        "Sensor Configuration:"
        # "- IMU: 2 Astrix-120 units near center-of-mass (low drift: 5e-4°/hr)\n"
        # "- Star Trackers: 2 Leonardo AA-STR units (FOV: 30°, accuracy: 8.25 arcsec)\n"
        # "- Fine Sun Sensors: 4 Redwire Adcole units (FOV: ±50°, accuracy: 0.01-0.05°)\n"
        # "- Coarse Sun Sensors: 6 Bradford Space units (FOV: 180°×180°, accuracy: ±2°)"
    )
    
    # Add text box for annotations
    props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
    ax.text2D(0.05, 0.05, annotation_text, transform=ax.transAxes, fontsize=18,
              verticalalignment='bottom', bbox=props, color='white')
    
    # Create custom legend
    imu_legend = mpatches.Patch(color='white', label='IMU')
    star_legend = mpatches.Patch(color='cyan', label='Star Trackers')
    fss_legend = mpatches.Patch(color='yellow', label='Fine Sun Sensors')
    # css_legend = mpatches.Patch(color='magenta', label='Coarse Sun Sensors')
    
    ax.legend(handles=[imu_legend, star_legend, fss_legend], 
              loc='upper left', facecolor='#333333', edgecolor='white', labelcolor='white', fontsize=18)
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()


def visualize_actuator_configuration(dimensions=[1.6, 1, 1.3]):
    """
    Visualize the actuator configuration for a cuboid spacecraft.
    
    Parameters:
    -----------
    dimensions : list
        Dimensions of the spacecraft [length, width, height] in meters.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Set global font size for the plot
    plt.rcParams.update({'font.size': 18})
    
    # Draw the cuboid spacecraft
    draw_cuboid(ax, dimensions)
    
    # Unpack dimensions for easier reference
    length, width, height = dimensions
    
    # Draw body axes
    draw_vector(ax, [0, 0, 0], [4.5, 0, 0], color='red', label="+X", arrow_length_ratio=1)
    draw_vector(ax, [0, 0, 0], [0, 4.5, 0], color='green', label="+Y", arrow_length_ratio=1)
    draw_vector(ax, [0, 0, 0], [0, 0, 4.5], color='blue', label="+Z", arrow_length_ratio=1)
    
    # 1. Honeywell HR16 Reaction Wheels in pyramidal configuration
    c = 2**(-0.5)  # This is approximately 0.7071 (1/√2)
    
    # Define reaction wheel positions (close to center of mass)
    rw_positions = [
        [0.2, 0, 0],   # RW1 position
        [0, 0.2, 0],   # RW2 position
        [-0.2, 0, 0],  # RW3 position
        [0, -0.2, 0]   # RW4 position
    ]
    
    # Define reaction wheel spin axes exactly as in spacecraft.py
    rw_axes = [
        [c, 0, c],      # RW1: [0.7071, 0, 0.7071] (+X, +Z)
        [0, c, c],      # RW2: [0, 0.7071, 0.7071] (+Y, +Z)
        [-c, 0, c],     # RW3: [-0.7071, 0, 0.7071] (-X, +Z)
        [0, -c, c]      # RW4: [0, -0.7071, 0.7071] (-Y, +Z)
    ]
    
    # Label for each wheel (corresponding to spacecraft.py)
    rw_labels = ["RW1 (+X,+Z)", "RW2 (+Y,+Z)", "RW3 (-X,+Z)", "RW4 (-Y,+Z)"]
    
    # Normalize the axes
    for i in range(len(rw_axes)):
        rw_axes[i] = np.array(rw_axes[i]) / np.linalg.norm(rw_axes[i])
    
    # Draw reaction wheels as disks
    for i, (pos, axis) in enumerate(zip(rw_positions, rw_axes)):
        # Draw the reaction wheel as a cylinder
        # First find perpendicular vectors to the axis for the cylinder
        perp1 = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, axis) * axis
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        
        # Create circle points for the rim of the wheel
        theta = np.linspace(0, 2*np.pi, 36)
        rim_x = []
        rim_y = []
        rim_z = []
        
        for t in theta:
            point = np.array(pos) + 0.15 * (perp1 * np.cos(t) + perp2 * np.sin(t))
            rim_x.append(point[0])
            rim_y.append(point[1])
            rim_z.append(point[2])
        
        # Close the circle
        rim_x.append(rim_x[0])
        rim_y.append(rim_y[0])
        rim_z.append(rim_z[0])
        
        # Draw the rim of the wheel
        ax.plot(rim_x, rim_y, rim_z, color='#00FFFF', linewidth=2)
        
        # Draw the axis of the wheel
        draw_vector(ax, pos, axis, color='#00FFFF', arrow_length_ratio=0.3)
        
        # Label the wheel
        ax.text(pos[0]+0.1, pos[1]+0.1, pos[2]+0.1, rw_labels[i], color='#00FFFF', fontsize=18)
    
    # 2. MOOG Monarch 90 HT Thrusters
    # Based on the image, place 12 thrusters (2 on each face)
    # Define thruster positions and directions according to the image
    thruster_positions = [
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
    
    # Define thruster directions (outward from the surface)
    thruster_directions = [
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
    
    # Draw thrusters
    for i, (pos, direction) in enumerate(zip(thruster_positions, thruster_directions)):
        thruster_id = i + 1  # Thruster number (1-12)
        
        # Draw thruster as an arrow
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color='#FF9900', s=80, marker='o')
        draw_vector(ax, pos, direction, color='#FF5500', arrow_length_ratio=0.3)
        
        # Label thruster
        ax.text(pos[0]+0.05*direction[0], 
                pos[1]+0.05*direction[1], 
                pos[2]+0.05*direction[2], 
                f"{thruster_id}", color='#FF9900', fontsize=18, weight='bold')
    
    # Add explanation for paired thrusters
    paired_thrusters = [
        "1, 2: Front face (-Y)",
        "3, 4: Top face (+Z)",
        "5, 6: Left side face (-X)",
        "7, 8: Top edges (±X, +Z)",
        "9, 10: Bottom edges (-Z)",
        "11, 12: Right side face (+X)"
    ]
    paired_text = "\n".join(paired_thrusters)
    
    # Set axis limits and labels
    max_dim = max(dimensions) * 1.8
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_zlim(-max_dim, max_dim)
    
    ax.set_xlabel('X (m)', color='white', fontsize=18)
    ax.set_ylabel('Y (m)', color='white', fontsize=18)
    ax.set_zlabel('Z (m)', color='white', fontsize=18)
    ax.set_title(f'Spacecraft Actuator Configuration - Dimensions: {dimensions} m', color='white', fontsize=18)
    
    # Set tick colors to white
    ax.tick_params(colors='white', labelsize=18)
    
    # Add annotations explaining the configuration
    annotation_text = (
        "Actuator Configuration:"
        # "- 4 Honeywell HR16 Reaction Wheels in pyramidal configuration:\n"
        # "  RW1: [√2/2, 0, √2/2] = [+X,+Z]\n"
        # "  RW2: [0, √2/2, √2/2] = [+Y,+Z]\n"
        # "  RW3: [-√2/2, 0, √2/2] = [-X,+Z]\n"
        # "  RW4: [0, -√2/2, √2/2] = [-Y,+Z]\n"
        # "  (Up to 0.4 Nm torque, 150 Nms momentum capacity)\n"
        # "- 12 MOOG Monarch 90 HT Thrusters (111 N thrust, 232 s Isp)\n"
        # "  " + paired_text
    )
    
    # Add text box for annotations
    props = dict(boxstyle='round', facecolor='#333333', alpha=0.8)
    ax.text2D(0.05, 0.05, annotation_text, transform=ax.transAxes, fontsize=18,
              verticalalignment='bottom', bbox=props, color='white')
    
    # Create custom legend
    rw_legend = mpatches.Patch(color='#00FFFF', label='Reaction Wheels')
    thruster_legend = mpatches.Patch(color='#FF9900', label='Thrusters')
    
    ax.legend(handles=[rw_legend, thruster_legend], loc='upper left',
              facecolor='#333333', edgecolor='white', labelcolor='white', fontsize=18)
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Visualize first spacecraft configuration
    small_spacecraft = [1, 1.3, 1.6]  # length x width x height in meters
    visualize_sensor_configuration(small_spacecraft)
    visualize_actuator_configuration(small_spacecraft)
    
    # Visualize second spacecraft configuration
    large_spacecraft = [2, 2.1, 2.8]  # length x width x height in meters
    visualize_sensor_configuration(large_spacecraft)
    visualize_actuator_configuration(large_spacecraft)

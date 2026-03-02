import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Replace with your actual file name
filename = "/content/drive/MyDrive/COS 526/gs_assignment/output/lego/train/ours_30000/fuse_post.ply"

# Load the point cloud
pcd = o3d.io.read_point_cloud(filename)
points = np.asarray(pcd.points)

# Set up the figure and 3D axes
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue')

# Method 1: Using Matplotlib 3.3+ set_box_aspect (if available)
try:
    ax.set_box_aspect((1, 1, 1))  # Ensures equal scaling
except Exception as e:
    print("set_box_aspect not available, falling back to custom method.")

    # Method 2: Helper function to set equal aspect ratio for older versions
    def set_axes_equal(ax):
        """Set equal scaling for 3D plots."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

# Label the axes and add a title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Point Cloud with Equal Axis Scaling")

plt.show()

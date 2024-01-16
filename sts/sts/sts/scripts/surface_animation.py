import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def get_gaussian_pcl(scale=1, num=25):
    x=np.linspace(-10,10, num=num)
    y=np.linspace(-10,10, num=num)
    x, y = np.meshgrid(x, y)
    z = scale * np.exp(-.01*x**2-.01*y**2)
    return np.dstack((x,y,z)).reshape(num**2, 3)

def interpolate_pcl(pcl_points, num=100, xrange=(-10,10), yrange=(-10,10), method='cubic'):
    x=np.linspace(xrange[0], xrange[1], num=num)
    y=np.linspace(yrange[0], yrange[1], num=num)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.array(pcl_points)[:,:2]
    values = np.array(pcl_points)[:,2]
    interpolated_values = griddata(points, values, (grid_x, grid_y), method=method)
    return np.nan_to_num(np.dstack((grid_x,grid_y,interpolated_values)).reshape(num**2, 3))

class SurfaceAnimation(object):
    def __init__(self, interpolate=False, visible=True):
        import open3d as o3d
        self.pcl = o3d.geometry.PointCloud()
        self.num_elements = 100
        # self.cmap = plt.get_cmap("cool")
        self.cmap = plt.get_cmap("gnuplot2")
        self.interpolate = interpolate
        self.visible = visible

    def _initialize(self, window_width=640, window_height = 480):
        import open3d as o3d
        self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        self.vis.create_window(width=window_width, height=window_height, left=1, top=1, visible=self.visible)
        xpoints = np.vstack((np.linspace(0, 480, 480),np.zeros(480),np.zeros(480))).transpose()
        self.vis.add_geometry(self.pcl)

        vc = self.vis.get_view_control()
        vc.rotate(0.0,-350.0)
        vc.set_zoom(0.55)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.1,0.1,0.1])

    def set_colors(self, points, max_value=15):
        import open3d as o3d
        colors = self.cmap(points[:,2]/np.amax(points[:,2]))[:,:3]
        # colors = self.cmap(points[:,2]/max_value)[:,:3]
        self.pcl.colors = o3d.utility.Vector3dVector(np.array(colors)[:,:3])

    def save_image(self, filename):
        self.vis.capture_screen_image(filename)

    def update_points(self, points):
        import open3d as o3d
        if self.interpolate:
            points = interpolate_pcl(points, num=100, xrange=(0,400), yrange=(0,400), method='cubic')
        if len(self.pcl.points)==0:
            self.pcl.points = o3d.utility.Vector3dVector(points)
            self._initialize()
        self.pcl.points = o3d.utility.Vector3dVector(points)
        self.set_colors(points)
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def __del__(self):
        if hasattr(self, 'vis'):
            self.vis.destroy_window()

def demo():
    animation = SurfaceAnimation()
    angle_list = np.linspace(0, 10*3.1416, 1000)
    for angle in angle_list:
        scale = 15 * np.sin(angle)
        gaussian_points = get_gaussian_pcl(scale=scale)
        interpolated_points = interpolate_pcl(gaussian_points)
        animation.update_points(interpolated_points)

if __name__ == "__main__":
    demo()

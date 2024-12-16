import numpy as np


class Hist3D:
    def __init__(self, limits, bin_size):
        self.limits = limits
        self.bin_size = bin_size
        self.p = None
        self.H = None
        self.X, self.Y, self.Z = None, None, None

    def set_params(self, limits, bin_size):
        self.limits = limits
        self.bin_size = bin_size
        return

    def create_centered_grid(self, p):
        if type(p) is not np.ndarray:
            self.p = np.asarray(p)
        else:
            self.p = p

        if len(self.limits) == 2:
            x_min, x_max = self.limits
            x_grid = np.arange(x_min, x_max + self.bin_size, self.bin_size).astype(np.float32)
            y_grid = np.copy(x_grid)
            z_grid = np.copy(x_grid)
        if len(self.limits) == 3:
            [x_min, x_max], [y_min, y_max], [z_min, z_max] = self.limits
            x_grid = np.arange(x_min, x_max + self.bin_size, self.bin_size).astype(np.float32)
            y_grid = np.arange(y_min, y_max + self.bin_size, self.bin_size).astype(np.float32)
            z_grid = np.arange(z_min, z_max + self.bin_size, self.bin_size).astype(np.float32)
        # Shift grid to center on p
        p_x, p_y, p_z = p
        self.X = self.move_grid_1d(x_grid, p_x)
        self.Y = self.move_grid_1d(y_grid, p_y)
        self.Z = self.move_grid_1d(z_grid, p_z)
        return

    def move_grid_1d(self, x_grid, p_x):
        # compute the closest distance to a given bin edge
        x_arg_closest_abs = np.argmin(np.abs(p_x - x_grid))
        x_closest = (p_x - x_grid)[x_arg_closest_abs]
        # Move grid onto the point (i.e. + x_closest) and then move half a bin width away (i.e. - bin_size/2)
        x_grid += (x_closest - np.float32(self.bin_size) / 2)
        return x_grid

    def create_hist(self, data):
        self.H, _ = np.histogramdd(data, bins=[self.X, self.Y, self.Z])
        return self.H

    def marginalize(self, dim2sum):
        # Get correct remaining axes
        all_axes = [0, 1, 2]
        all_axes.remove(dim2sum)
        # Get remaining axes limits (as extents)
        xyz = [self.X, self.Y, self.Z]
        # Marginalize over axis & prepare correct extents
        H_ij = np.sum(self.H, axis=dim2sum)
        extents = [xyz[i][j] for i in all_axes for j in [0, -1]]
        return H_ij, extents

    def hist2d(self, data, ref_pt, dim2sum, plt_axis, density=False, **kwargs):
        self.create_centered_grid(ref_pt)
        self.create_hist(data)
        H_ij, extents = self.marginalize(dim2sum)
        if density:
            H_ij = H_ij / np.sum(H_ij)
        im_plt = plt_axis.imshow(H_ij.T, extent=extents, origin='lower', **kwargs)
        return im_plt

    def select_voxels_within_sphere(self, radius):
        # 1. Calculate bin centers from edges
        x_centers = (self.X[:-1] + self.X[1:]) / 2
        y_centers = (self.Y[:-1] + self.Y[1:]) / 2
        z_centers = (self.Z[:-1] + self.Z[1:]) / 2
        # 2. Create meshgrid
        Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        # 3. Compute distance from center
        dist = np.sqrt((Xc - self.p[0]) ** 2 + (Yc - self.p[1]) ** 2 + (Zc - self.p[2]) ** 2)
        # 4. Select voxels within sphere
        bool_mask = dist <= radius
        return bool_mask

    def voxel_counts(self, radius, data=None, ref_pt=None):
        if data is None:
            if self.H is None:
                raise ValueError('No histogram has been created yet. Please provide data.')
        else:
            self.create_centered_grid(ref_pt)
            self.create_hist(data)

        bool_mask = self.select_voxels_within_sphere(radius)
        return self.H[bool_mask]

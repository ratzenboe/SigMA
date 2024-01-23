import copy
import pandas as pd
from scipy.spatial import KDTree


class DataLayer:
    def __init__(
        self,
        data: pd.DataFrame,
        cluster_features: list,
        scale_factors: dict = None,
        **kwargs
    ):
        """Class calculating densities on given data X
        data: pandas data frame containing all necessary columns
        cluster_features: Features used in the clustering process
        scale_factors: Features that are scaled with given factors, needs specific layout:
            scale_factors: {
                    pos: {'features': ['v_alpha', 'v_delta'], 'factor': 5}
            }

        """
        self.data = data
        self.cluster_columns = cluster_features
        self.scale_factors = scale_factors
        self.X = self.init_cluster_data()
        self.kd_tree = KDTree(data=self.X)
        # Meta data
        self.meta_pos = None
        self.meta_vel = None
        self.auto_infer_feature_space()

    def update_scaling_factors(self, scale_factors: dict):
        """Change scale factors and re-initialize clustering"""
        self.scale_factors = scale_factors
        # Update data and kd-tree
        self.X = self.init_cluster_data()
        self.kd_tree = KDTree(data=self.X)
        return self

    def init_cluster_data(self):
        X = copy.deepcopy(self.data[self.cluster_columns])
        if self.scale_factors is not None:
            for scale_info in self.scale_factors.values():
                cols = scale_info["features"]
                sf = scale_info["factor"]
                X[cols] *= sf
        return X.values

    def auto_infer_feature_space(self):
        cols = [col.lower() for col in self.data.columns]
        # Spatial information
        if ("x" in cols) or ("y" in cols) or ("z" in cols):
            self.meta_pos = "cartesian"
        elif ("ra" in cols) or ("dec" in cols) or ("parallax" in cols):
            self.meta_pos = "spherical"
        else:
            print(
                "Positional information cannot be inferred from data frame! "
                "User must pass transformation function for resampling!"
            )
        # Velocity information
        if (
            ("v_x" in cols)
            or ("v_y" in cols)
            or ("v_z" in cols)
            or ("u" in cols)
            or ("v" in cols)
            or ("w" in cols)
        ):
            self.meta_vel = "cartesian"
        elif ("pmra" in cols) or ("pmdec" in cols) or ("radial_velocity" in cols):
            self.meta_vel = "spherical"
        elif (
            ("v_alpha" in cols)
            or ("v_delta" in cols)
            or ("v_a_lsr" in cols)
            or ("v_d_lsr" in cols)
        ):
            self.meta_vel = "spherical_lsr"
        else:
            print(
                "Velocity information cannot be inferred from data frame! "
                "User must pass transformation function for resampling!"
            )
        return self

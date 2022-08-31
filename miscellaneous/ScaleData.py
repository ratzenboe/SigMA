import numpy as np
import pandas as pd
import copy


class ScaleData:
    def __init__(self, data: pd.DataFrame, scaling_factors: list, individual=False, use_copy: bool = False):
        """
        :param data: data to be scaled
        :param scaling_factors: should look like this

             scaling_factors = [{'axes': ['X', 'Y', 'Z'], 'scale': 1},
                                {'axes': ['v_alpha', 'v_delta'], 'scale': 1.25}
                               ]
        """
        if use_copy:
            self.data = copy.deepcopy(data)
        else:
            self.data = data
        self.scale_individually = individual
        sf = copy.deepcopy(scaling_factors)
        # Calculate average standard deviation
        for axes_scale_dict in sf:
            cols = axes_scale_dict['axes']
            std = data[cols].std(axis=0)
            if individual:
                axes_scale_dict['avg_std'] = std
            else:
                axes_scale_dict['avg_std'] = np.mean(std.values)
        # store scaling information
        self.scaling_info = sf

    def scale_input(self, X=None):
        """
        :param data: Scale input data with scaling_factors and avg_std
        """
        if X is None:
            X = self.data
        # Scale data accordingly
        for axes_scale_dict in self.scaling_info:
            cols, scale_factor, avg_std = axes_scale_dict['axes'], axes_scale_dict['scale'], axes_scale_dict['avg_std']
            X[cols] /= avg_std  # scale multiple axes as a whole and not individually
            X[cols] *= scale_factor  # scle axes by scale_factor
        return X
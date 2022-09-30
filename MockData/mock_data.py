import numpy as np
import pandas as pd
import copy
from MockData.affine_transformations import translate_points, rotate_points_3d
from NoiseRemoval.TransformCoordinates import transform_inverse_UVW, transform_inverse_XYZ
from miscellaneous.utils import isin_range
from astropy.coordinates import LSR, SkyCoord, Distance
import astropy.units as u


class MockData:
    def __init__(self, data, labels, pos_axes, vel_axes):
        self.pos_axes = pos_axes
        self.vel_axes = vel_axes
        self.mock_info = self.compute_cluster_infos(data, labels)

    def compute_cluster_infos(self, data, labels):
        """Compute mean positions and disperions """
        mock_info = {}
        all_axes = self.pos_axes + self.vel_axes
        for ul in np.unique(labels):
            cut = labels == ul
            mean_info_ul = {col: m for m, col in zip(data.loc[cut, all_axes].mean().values, all_axes)}
            mock_info[ul] = mean_info_ul
            mock_info[ul]['nb_stars'] = np.sum(cut)
            # Compute median absolute deviation in v_T space (to scale proper motions later)
            diff = (data.loc[cut, self.vel_axes] - data.loc[cut, self.vel_axes].mean()).values
            diff = np.linalg.norm(diff, axis=1)
            mock_info[ul]['mad_vel'] = np.median(diff)
            # Same in XYZ space
            diff = (data.loc[cut, self.pos_axes] - data.loc[cut, self.pos_axes].mean()).values
            diff = np.linalg.norm(diff, axis=1)
            mock_info[ul]['mad_pos'] = np.median(diff)
        return mock_info

    def randomize_star_count(self, mock_sizes, min_members=30):
        """Produce a random star count for each cluster"""
        for scocen_uc, scocen_info in self.mock_info.items():
            nb_sc = scocen_info['nb_stars']
            # select clusters to represent current cluster
            # Probability proportional to size differences
            p = 1 / (nb_sc - mock_sizes) ** 2
            p /= np.sum(p)
            mock_cluster_representative = np.random.choice(p.size, p=p, size=1)[0]
            scocen_info['mock_representative'] = mock_cluster_representative
            # Determine number of mock stars: target number of stars should be about 10% more or less
            new_nb_points = min(int(np.random.normal(nb_sc, nb_sc / 10, 1)[0]), mock_sizes[mock_cluster_representative])
            scocen_info['nb_mock'] = max(new_nb_points, min_members)
        return

    def build_sample(self, data_mock, labels, pos_axes, vel_axes):
        """Generate new data set sample"""
        data_idx = np.arange(data_mock.shape[0])
        # Create new mock catalog
        for scocen_uc, scocen_info in self.mock_info.items():
            # Determine mean positions in phase space of ScoCen clusters
            mean_xyz = np.array([
                scocen_info[self.pos_axes[0]],
                scocen_info[self.pos_axes[1]],
                scocen_info[self.pos_axes[2]]
            ])
            mean_vel = np.array([
                scocen_info[self.vel_axes[0]],
                scocen_info[self.vel_axes[1]],
                scocen_info[self.vel_axes[2]]
            ])
            # Get mock representative & sample from it
            mock_rep = scocen_info['mock_representative']
            nb_mock = scocen_info['nb_mock']

            # --- Data in feature space ---
            mock_cluster = copy.deepcopy(data_mock.loc[labels == mock_rep, pos_axes + vel_axes])
            # -- Scale PMs --
            # Compute MAP in pm space
            diff = (mock_cluster[vel_axes] - mock_cluster[vel_axes].mean()).values
            diff = np.linalg.norm(diff, axis=1)
            mad_pm = np.median(diff)
            # Get random value around orginal value (10% variance)
            mad_orig = np.random.normal(scocen_info['mad_vel'], scocen_info['mad_vel'] / 10, 1)[0]
            # Scale proper motion axes
            mock_cluster[vel_axes] *= mad_orig / mad_pm
            # -- Scale XYZ --
            # Compute MAP in XYZ space
            diff = (mock_cluster[pos_axes] - mock_cluster[pos_axes].mean()).values
            diff = np.linalg.norm(diff, axis=1)
            mad_xyz = np.median(diff)
            # Get random value around orginal value (10% variance)
            mad_orig = np.random.normal(scocen_info['mad_pos'], scocen_info['mad_pos'] / 10, 1)[0]
            # Scale proper motion axes
            mock_cluster[pos_axes] *= mad_orig / mad_xyz

            # --- Get random subsample ---
            a = np.arange(mock_cluster.shape[0])
            subsample = np.random.choice(a, size=nb_mock, replace=False)
            is_new_cluster = np.isin(a, subsample)
            mock_cluster = mock_cluster.loc[is_new_cluster].reset_index(drop=True)
            # -- Move clusters to mean positions --
            pos_data = mock_cluster[pos_axes].values
            vel_data = mock_cluster[vel_axes].values
            # translate points
            pos_data = translate_points(pos_data, mean_xyz)
            vel_data = translate_points(vel_data, mean_vel)
            # -- Rotate data in random diration --
            # XYZ
            vector = np.random.normal(loc=0.0, scale=1.0, size=3)
            angle = np.random.random() * 360
            pos_data = rotate_points_3d(pos_data, vector=vector, angle=angle)
            # vel
            vector = np.random.normal(loc=0.0, scale=1.0, size=3)
            angle = np.random.random() * 360
            vel_data = rotate_points_3d(vel_data, vector=vector, angle=angle)
            # Transform data to observables
            ra_dec_plx = transform_inverse_XYZ(pos_data)
            ra, dec, plx = ra_dec_plx.T
            pmra_pmdec_rv = transform_inverse_UVW(vel_data, ra, dec, plx)
            # --- save data ---
            df = pd.DataFrame(
                np.concatenate([pos_data, vel_data, ra_dec_plx, pmra_pmdec_rv], axis=1),
                columns=self.pos_axes + self.vel_axes + ['ra', 'dec', 'parallax'] + ['pmra', 'pmdec', 'radial_velocity']
            )
            df['labels'] = scocen_uc
            # Remove already given columns from list
            columns = set(list(data_mock.columns)) - set(list(df.columns))
            # Get remaining columns
            df_rest = data_mock.loc[
                np.isin(data_idx, data_idx[labels == mock_rep][is_new_cluster]),
                list(columns)
            ].reset_index(drop=True)
            df = pd.concat([df, df_rest], axis=1)
            scocen_info['mock_data'] = df
        return

    def new_sample(self, data_mock, labels, pos_axes, vel_axes, min_members=30):
        """Create a new mock catalog"""
        mock_sizes = np.array([np.sum(data_mock.labels == uc) for uc in np.unique(labels)])
        # Step 1: update number of stars per cluster
        self.randomize_star_count(mock_sizes, min_members)
        # Step 2: create new cluster catalog
        self.build_sample(data_mock, labels, pos_axes, vel_axes)
        # Step 3: Concatenate to single catalog with labels
        df_mock = pd.concat([sc_info['mock_data'] for sc_info in self.mock_info.values()], ignore_index=True)
        return df_mock

    def embed_in_bg(self, data_clusters, data_bg, features):
        if ('labels' not in features) or ('labels' not in data_clusters.columns):
            raise KeyError('"labels" not provided')

        data_bg['labels'] = -1
        df = pd.concat([data_clusters[features], data_bg[features]], ignore_index=True)
        return df.reset_index(drop=True)

    def next(self, data_mock, labels, pos_axes, vel_axes, data_bg, features, min_members=30, limits=None):
        df_new = self.new_sample(data_mock, labels, pos_axes, vel_axes, min_members=min_members)
        df_new = df_new[features]
        df_mix = self.embed_in_bg(data_clusters=df_new, data_bg=data_bg, features=features)
        # Add tangential velocities in lsr
        skycoord = SkyCoord(
            ra=df_mix['ra'].values * u.deg,
            dec=df_mix['dec'].values * u.deg,  # 2D on sky postition
            distance=Distance(parallax=df_mix['parallax'].values * u.mas),  # distance in pc
            pm_ra_cosdec=df_mix['pmra'].values * u.mas / u.yr,
            pm_dec=df_mix['pmdec'].values * u.mas / u.yr,
            radial_velocity=0. * u.km / u.s,
            frame="icrs"
        )
        # Transform to lsr
        pma_lsr = skycoord.transform_to(LSR()).pm_ra_cosdec.value
        pmd_lsr = skycoord.transform_to(LSR()).pm_dec.value
        df_mix['v_a_lsr'] = 4.74047 * pma_lsr / df_mix['parallax'].values
        df_mix['v_d_lsr'] = 4.74047 * pmd_lsr / df_mix['parallax'].values

        # Remove sources if not in range
        inrange = np.ones(df_mix.shape[0], dtype=bool)
        if limits is not None:
            for col, lim in limits.items():
                inrange &= isin_range(df_mix, col, lim).values

        return df_mix.loc[inrange]
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance
from astropy.coordinates import ICRS, GalacticLSR
from astropy.time import Time
import astropy.units as u
from astropy.units import dimensionless_angles
import math


class OpenData:
    """Class to open and manipulate data"""

    def __init__(self, fpath: str = None, data: pd.DataFrame = None):
        self.data = data
        self.fpath = fpath
        self.skycoord = None
        self.galactic_lsr = None

    def import_data(self, fpath=None):
        """Import data from either .fits or .pkl files
        :param data_path: Pickle of fits file path
        :return: Pandas data frame
        """
        if fpath is None:
            fpath = self.fpath

        suffix_readfunc_dict = {'.pkl': pd.read_pickle,
                                '.fits': lambda x: Table.read(x).to_pandas(),
                                '.csv': pd.read_csv}
        for suffix, func in suffix_readfunc_dict.items():
            if fpath.endswith(suffix):
                self.data = func(fpath)
        return self.data

    def rename_columns(self, ra='ra', dec='dec', parallax='parallax',
                       pmra='pmra', pmdec='pmdec', rv='radial_velocity'):
        """Rename columns to fit internal naming convention"""
        rename_dict = {ra: 'ra', dec: 'dec', parallax: 'parallax',
                       pmra: 'pmra', pmdec: 'pmdec', rv: 'radial_velocity'
                       }
        self.data = self.data.rename(columns=rename_dict)
        if 'mag_g' not in self.data.columns:
            self.data['mag_g'] = self.data['phot_g_mean_mag'] + 5 * self.data['parallax'].apply(math.log10) - 10
        return

    def create_skycoords(self):
        # --- Observation time ---
        if 'ref_epoch' not in self.data.columns:
            self.data['ref_epoch'] = 2016
        obstime = Time(self.data['ref_epoch'].values, format='decimalyear')
        # --- Distance ---
        if 'r_est' in self.data.columns:
            distance = Distance(self.data['r_est'].values, unit=u.pc)
        else:
            distance = Distance(parallax=self.data['parallax'].values * u.mas)

        self.skycoord = SkyCoord(
            ra=self.data['ra'].values * u.deg, dec=self.data['dec'].values * u.deg,  # 2D on sky postition
            distance=distance,  # distance in pc
            pm_ra_cosdec=self.data['pmra'].values * u.mas / u.yr,  # PMRA
            pm_dec=self.data['pmdec'].values * u.mas / u.yr,  # RMDec
            frame="icrs", obstime=obstime,  # Coordinate frame and obs-time
            radial_velocity=self.data['radial_velocity'].values * u.km / u.s  # radial veclocity
        )
        icrs = ICRS(ra=self.data['ra'].values * u.deg,
                    dec=self.data['dec'].values * u.deg,
                    distance=distance,
                    pm_ra_cosdec=self.data['pmra'].values * u.mas / u.yr,  # PMRA
                    pm_dec=self.data['pmdec'].values * u.mas / u.yr,  # RMDec
                    radial_velocity=self.data['radial_velocity'].values * u.km / u.s
                    )

        # transform to Galactic frame with velocties in LSR frame
        self.galactic_lsr = icrs.transform_to(GalacticLSR)
        self.galactic_lsr.representation_type = 'cartesian'  # set positions to cartesian
        self.galactic_lsr.differential_type = 'cartesian'  # set velocities to cartesian

        return

    def set_XYZ(self):
        if self.skycoord is None:
            raise ValueError('Skycoors must be created first')
        # Set x,y,z positions
        self.data['X'] = self.galactic_lsr.x.value
        self.data['Y'] = self.galactic_lsr.y.value
        self.data['Z'] = self.galactic_lsr.z.value
        return

    def set_UVW(self):
        if self.skycoord is None:
            raise ValueError('Skycoors must be created first')
        # Set galactic, Cartesian vx,vy,vz velocities
        self.data['U'] = self.galactic_lsr.v_x.value
        self.data['V'] = self.galactic_lsr.v_y.value
        self.data['W'] = self.galactic_lsr.v_z.value
        return

    def set_cylindrical_velocities(self):
        if self.skycoord is None:
            raise ValueError('Skycoors must be created first')
        # Cylindric coordinates
        # Calculation of cylindrical variables
        skycoord_cyl = self.skycoord.galactocentric
        skycoord_cyl.representation_type = 'cylindrical'
        vr = skycoord_cyl.d_rho.to('km/s')
        dummy = skycoord_cyl.rho * - skycoord_cyl.d_phi
        vphi = dummy.to('km/s', dimensionless_angles())
        vz = skycoord_cyl.d_z.to('km/s')
        # Save the cylindrical vel. info in the training set: vr_cylinder, vphi_cylinder, vz_cylinder
        self.data['vr_cylinder'] = vr.value
        self.data['vphi_cylinder'] = vphi.value
        self.data['vz_cylinder'] = vz.value
        return

    def set_tangential_velocities(self):
        if self.skycoord is None:
            raise ValueError('Skycoors must be created first')
        # Transform proper motion into tangential velocities
        self.data['v_alpha'] = 4.74047 * self.data.pmra / self.data.parallax
        self.data['v_delta'] = 4.74047 * self.data.pmdec / self.data.parallax
        return

    def set_velocities(self):
        if self.skycoord is None:
            raise ValueError('Skycoors must be created first')
        # --- Set all velocities ---
        self.set_UVW()
        self.set_cylindrical_velocities()
        self.set_tangential_velocities()
        return



def csv2pandas(data_info):
    """
    Usage example:
    data_dict = {
        'skinny': {
            'path': '/home/sebastian/Documents/PhD/data/skinnydip/skinnyDipData_8.csv',
            'read_csv_kwargs': {},
            'columns2keep': ['V2', 'V3', 'V4'],
            'rename_columns': {'V2': 'f1', 'V3': 'f2', 'V4': 'labels'},
            'astype': {'f1': np.float32, 'f2': np.float32, 'labels': np.int32},
        },
        'adawave': {
            'path': '/home/sebastian/Documents/PhD/data/adawave/waveData_8.csv',
            'read_csv_kwargs': dict(header=None, names=['f1', 'f2', 'labels']),
            'columns2keep': ['f1', 'f2', 'labels'],
            'rename_columns': {},
            'astype': {'f1': np.float32, 'f2': np.float32, 'labels': np.int32}
        }
    }
    """
    X = pd.read_csv(data_info['path'], **data_info['read_csv_kwargs'])
    X = X[data_info['columns2keep']]
    X = X.rename(columns=data_info['rename_columns'])
    X = X.dropna()
    X = X.astype(data_info['astype'])
    return X, data_info['columns2keep']

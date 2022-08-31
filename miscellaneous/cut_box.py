import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u
from astropy.units import dimensionless_angles
import math


def import_data(fpath):
    """Import data from either .fits or .pkl files
    :param data_path: Pickle of fits file path
    :return: Pandas data frame
    """
    data = None
    suffix_readfunc_dict = {'.pkl': pd.read_pickle,
                            '.fits': lambda x: Table.read(x).to_pandas(),
                            '.csv': pd.read_csv}
    for suffix, func in suffix_readfunc_dict.items():
        if fpath.endswith(suffix):
            data = func(fpath)
    return data


def create_skycoords(data):
    # --- Observation time ---
    if 'ref_epoch' not in data.columns:
        data['ref_epoch'] = 2016
    obstime = Time(data['ref_epoch'].values, format='decimalyear')
    # --- Distance ---
    if 'r_est' in data.columns:
        distance = Distance(data['r_est'].values, unit=u.pc)
    else:
        distance = Distance(parallax=data['parallax'].values * u.mas)
    skycoord = SkyCoord(
        ra=data['ra'].values * u.deg, dec=data['dec'].values * u.deg,  # 2D on sky postition
        distance=distance,  # distance in pc
        pm_ra_cosdec=data['pmra'].values * u.mas / u.yr,  # PMRA
        pm_dec=data['pmdec'].values * u.mas / u.yr,  # RMDec
        frame="icrs", obstime=obstime,  # Coordinate frame and obs-time
        radial_velocity=data['radial_velocity'].values * u.km / u.s  # radial veclocity
    )
    return skycoord


def set_XYZ(data, skycoord):
    # Set x,y,z positions
    data['X'] = skycoord.galactic.cartesian.x.value
    data['Y'] = skycoord.galactic.cartesian.y.value
    data['Z'] = skycoord.galactic.cartesian.z.value
    return data


def set_UVW(data, skycoord):
    # Set galactic, Cartesian vx,vy,vz velocities
    data['u'] = skycoord.galactic.velocity.d_x.value
    data['v'] = skycoord.galactic.velocity.d_y.value
    data['w'] = skycoord.galactic.velocity.d_z.value
    return data


def set_cylindrical_velocities(data, skycoord):
    # Cylindric coordinates
    # Calculation of cylindrical variables
    skycoord_cyl = skycoord.galactocentric
    skycoord_cyl.representation_type = 'cylindrical'
    vr = skycoord_cyl.d_rho.to('km/s')
    dummy = skycoord_cyl.rho * - skycoord_cyl.d_phi
    vphi = dummy.to('km/s', dimensionless_angles())
    vz = skycoord_cyl.d_z.to('km/s')
    # Save the cylindrical vel. info in the training set: vr_cylinder, vphi_cylinder, vz_cylinder
    data['vr_cylinder'] = vr.value
    data['vphi_cylinder'] = vphi.value
    data['vz_cylinder'] = vz.value
    return data


def set_tangential_velocities(data):
    # Transform proper motion into tangential velocities
    data['v_alpha'] = 4.74047 * data.pmra / data.parallax
    data['v_delta'] = 4.74047 * data.pmdec / data.parallax
    return data


def main():
    box_coords = {
        'X': [0, 120],
        'Y': [-200, -70],
        'Z': [-70, 0]
    }
    fname = None
    fname_save = None
    if fname is not None:
        data = import_data(fname)
        data['mag_abs_g'] = data['phot_g_mean_mag'] + 5 * data['parallax'].apply(math.log10) - 10
        skycoods = create_skycoords(data)
        data = set_XYZ(data, skycoods)
        data = set_UVW(data, skycoods)
        # Remove stars outside of box
        is_outside_box = np.zeros(data.shape[0], dtype=bool)
        for axis, axis_range in box_coords.items():
            is_outside_box |= (data[axis].values < np.min(axis_range))
            is_outside_box |= (data[axis].values > np.max(axis_range))
        data = data.loc[~is_outside_box]
        # Save data
        if fname_save is not None:
            data.to_csv(fname_save)
    return


if __name__ == "__main__":
    main()


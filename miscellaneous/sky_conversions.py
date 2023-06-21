from astropy.coordinates import SkyCoord, Distance
import astropy.units as u


def fill_galactic(data):
    skycoord = SkyCoord(ra=data['ra'].values * u.deg,
                        dec=data['dec'].values * u.deg,
                        distance=Distance(parallax=data['parallax'].values * u.mas),
                        pm_ra_cosdec=data['pmra'].values * u.mas/u.yr,
                        pm_dec=data['pmdec'].values * u.mas/u.yr,
                        radial_velocity=data['radial_velocity'].values * u.km/u.s,
                        frame='icrs')

    data['X'] = skycoord.galactic.cartesian.x.value
    data['Y'] = skycoord.galactic.cartesian.y.value
    data['Z'] = skycoord.galactic.cartesian.z.value

    data['U_lsr'] = skycoord.galacticlsr.velocity.d_x.value
    data['V_lsr'] = skycoord.galacticlsr.velocity.d_y.value
    data['W_lsr'] = skycoord.galacticlsr.velocity.d_z.value

    return data
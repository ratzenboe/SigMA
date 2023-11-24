from astropy.coordinates import LSR, SkyCoord, Distance
import astropy.units as u
import pandas as pd


def to_SkyCoord(ra, dec, parallax, pmra, pmdec):
    # Transform to different coordinate system
    skycoord = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,  # 2D on sky postition
        distance=Distance(parallax=parallax * u.mas),  # distance in pc
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        frame="icrs",
    )
    return skycoord


def gal_vtan_lsr(skycoord):
    x = skycoord.galactic.cartesian.x.value
    y = skycoord.galactic.cartesian.y.value
    z = skycoord.galactic.cartesian.z.value
    # Transform to lsr
    pma_lsr = skycoord.transform_to(LSR()).pm_ra_cosdec.value
    pmd_lsr = skycoord.transform_to(LSR()).pm_dec.value
    v_a_lsr = 4.74047 * pma_lsr / (1000 / skycoord.distance.value)
    v_d_lsr = 4.74047 * pmd_lsr / (1000 / skycoord.distance.value)
    df = pd.DataFrame(
        {
            "X": x,
            "Y": y,
            "Z": z,
            "v_a_lsr": v_a_lsr,
            "v_d_lsr": v_d_lsr,
        }
    )
    return df


# See possible tranfomation functions as input for SigMA
def transform_sphere_to_cartesian(ra, dec, parallax, pmra, pmdec):
    skycoord = to_SkyCoord(ra, dec, parallax, pmra, pmdec)
    df = gal_vtan_lsr(skycoord)
    return df


def idenity_transform(ra, dec, parallax, pmra, pmdec):
    df = pd.DataFrame(
        {
            "ra": ra,
            "dec": dec,
            "parallax": parallax,
            "pmra": pmra,
            "pmdec": pmdec,
        }
    )
    return df

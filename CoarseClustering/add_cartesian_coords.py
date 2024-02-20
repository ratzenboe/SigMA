import pandas as pd
from coordinate_transformations.sky_convert import transform_sphere_to_cartesian

from astropy.table import Table

# load the dataframe
df_focus = pd.read_csv('../Data/Gaia/Orion_Taurus_DR3_box_2022-12-12_4580041_FID_0p5_SNR_5_RAVE_GALAH.csv',
                       low_memory=False)

df_gal_coords = transform_sphere_to_cartesian(df_focus.ra.to_numpy(), df_focus.dec.to_numpy(),
                                              df_focus.parallax.to_numpy(), df_focus.pmra.to_numpy(),
                                              df_focus.pmdec.to_numpy())

df_fin = pd.concat([df_focus, df_gal_coords], axis=1)
t = Table.from_pandas(df_fin)

# df_fin.to_csv('../Data/Gaia/Orion_Taurus_full_galCoords_19-2-24.csv')
# t.write('../Data/Gaia/Orion_Taurus_full_galCoords_19-2-24.fits')

# ------------------------
# in case we also add RVs

# from coordinate_transformations.sky_convert import icrs2lsr2cart
# import numpy as np
# rv_nonan = np.nan_to_num(df_focus.radial_velocity.to_numpy(), nan=0)
# X, Y, Z, va_lsr, vd_lsr = icrs2lsr2cart([df_focus.ra.to_numpy(), df_focus.dec.to_numpy(),
#                                          df_focus.parallax.to_numpy(),
#                                          df_focus.pmra.to_numpy(), df_focus.pmdec.to_numpy(), rv_nonan])

# print("X:", min(X), max(X), "\n Y:", min(Y), max(Y), "\n Z:", min(Z), max(Z), "\n v_alpha_LSR:", min(va_lsr),
#      max(va_lsr), " \n v_delta_LSR:", min(vd_lsr), max(vd_lsr))
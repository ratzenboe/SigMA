from astropy.table import Table
from astropy.io import fits

hdulist = fits.open('../Data/Gaia/Orion_Taurus_DR3_box_2022-12-12_4580041_FID_0.5_SNR_5+RAVE+GALAH.fits')
t = Table.read(hdulist[1])

df = t.to_pandas()
print(df.head())
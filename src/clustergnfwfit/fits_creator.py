from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

# ref_ra, ref_dec in degrees
def make_fits(fpath, ref_ra, ref_dec, data, di_90, sigma_90, di_150, sigma_150):
    OBJ_NAME = 'macsj0025.4'
    PIX_SIZE = float(4) # length of pixel size (arcseconds)
    REDSHIFT = float(0.580000)
    R2500 = 1.11184
    R2500UNI = 'arcmin'
    XRAYRA0 = 6.37458333333
    XRAYDEC0 = -12.3791666667
    TXRAYPW = 8.31262
    TPWERR = 1.12403
    TXRAYMW = 8.02958
    TMWERR = 1.07158
    PLWDI = 0.0552723
    PMWDI = 0.00971516 


    hdr = fits.Header()
    
    str_date, date_comment = datetime.today().strftime('%Y-%m-%d'), "Creation UTC (CCCC-MM-DD)"
    hdr['DATE'] = (str_date, date_comment)

    hdr['PRIMARY'] = 'Best-fit gNFW model'

    hdr['OBJNAME'] = OBJ_NAME
    
    hdr['EQUINOX'] = float(2000)
    
    hdr['PIXSIZE'] = (PIX_SIZE, "length of pixel side (arcseconds)")

    hdr['TTYPE'] = 'R2500 normalized'

    hdr['TUNIT'] = 'keV'

    hdr['ACTDI1'] = di_90
    hdr['ACTDI2'] = di_150
    hdr['ACTERR1'] = sigma_90
    hdr['ACTERR2'] = sigma_150

    hdr['ACTUNIT'] = 'MJy/sr'

    hdr['REDSHIFT'] = REDSHIFT

    hdr['R2500'] = R2500

    hdr['R2500UNI'] = R2500UNI

    hdr['XRAYRA0'] = XRAYRA0
    hdr['XRAYDEC0'] = XRAYDEC0
    hdr['TXRAYPW'] = TXRAYPW
    hdr['TPWERR'] = TPWERR
    hdr['TXRAYMW '] = TXRAYMW
    hdr['TMWERR'] = TMWERR

    hdr['ACTNU01'] = (90, 'GHz')
    hdr['ACTNU02'] = (150, 'GHz')

    hdr['PLWDI'] = PLWDI
    hdr['PMWDI'] = PMWDI

    hdr['CTYPE1'] = ('RA---SFL', "Coordinate Type")
    hdr['CTYPE2'] = ('DEC---SFL', "Coordinate Type")

    hdr['CD1_1'] = (-0.00111111, "Degrees / Pixel")
    hdr['CD1_2'] = (-0.00000, "Degrees / Pixel")
    hdr['CD2_1'] = (0.00000, "Degrees / Pixel")
    hdr['CD2_2'] = (0.00111111, "Degrees / Pixel")

    hdr['CRPIX1'] = (235.500, "Reference Pixel in X")
    hdr['CRPIX2'] = (235.500, "Reference Pixel in Y")

    hdr['CRVAL1'] = (ref_ra, "R.A. (degrees) of reference pixel")
    hdr['CRVAL2'] = (ref_dec, "Declination of reference pixel")
    
    hdr['PV1_0'] = (0.00000000000, "Projection parameters")
    hdr['PV1_1'] = (0.00000000000, "Projection parameters")
    hdr['PV1_2'] = (0.00000000000, "Projection parameters")
    hdr['PV1_3'] = (180.000000000, "Projection parameters")
    hdr['PV1_4'] = (90.000000000, "Projection parameters")
    
    hdr['RADESYS'] = ('FK5', "Reference frame")

    primary = fits.PrimaryHDU(data=data, header=hdr)
    hdul = fits.HDUList([primary])
    hdul.writeto(fpath, overwrite=True)

import astropy.units as u
from astropy import coordinates as coords
from astroquery.sdss import SDSS

co = coords.SkyCoord(ra=194.95 * u.deg, dec=27.98 * u.deg, frame="icrs")


xid = SDSS.query_region(co, spectro=True, radius=2 * u.arcsec)

if xid:

    spec = SDSS.get_spectra(matches=xid)
    if spec:

        spec[0].writeto("galaxy_sample.fits", overwrite=True)
        print("Downloaded SDSS spectrum successfully!")
    else:
        print("No spectrum found for the given coordinates.")
else:
    print("No matching spectra found in the specified region.")

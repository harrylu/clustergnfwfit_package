# from astropy import units as u
# from astropy.cosmology import Planck15

# def convert_microkelvin_to_mjysr(map, freq):
#   freq = freq * u.GHz
#   equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
#   return ((map) * u.uK).to(u.MJy / u.sr, equivalencies=equiv).value
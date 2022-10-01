from astropy import units as u
from astropy.cosmology import Planck15

def convert_microkelvin_to_mjysr(map, freq):
  freq = freq * u.GHz
  equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
  # divide map by 1000 to convert to from microKelvin to milliKelvin
  return ((map / 1000) * u.mK).to(u.MJy / u.sr, equivalencies=equiv).value
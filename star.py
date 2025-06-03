# Roos Uit de Weerd
# Bachelor Project

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from scipy.optimize import fsolve

# initial conditions WR140
M_s = 1.988 * 10**30                                            # unit: kg
M_0_O = 1.8 * 10**-6 * M_s                                      # unit: kg/year     source: https://iopscience.iop.org/article/10.1086/431193/pdf
M_0_wr = 5.7 * 10**-5 * M_s                                     # unit: kg/year     source: https://iopscience.iop.org/article/10.1086/431193/pdf
v_O = 3200                                                      # unit: km/s        source: https://iopscience.iop.org/article/10.1086/431193/pdf
v_wr = 2860                                                     # unit: km/s        source: https://iopscience.iop.org/article/10.1086/431193/pdf
D_AU = 13.55                                                    # unit: AU          source: https://arxiv.org/pdf/2101.10563
D_mas = 8.922 * u.mas                                           # unit: mas         source: https://arxiv.org/pdf/2101.10563
alpha = (360 - 152 + 90) * u.deg                                # unit: deg         source: https://arxiv.org/pdf/2101.10563
c = SkyCoord('20h20m27.9757908696s', '+43d51m16.286840244s')    # unit: h and deg   source: https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=WR140&submit=SIMBAD+search
alpha_wcr = c.ra
delta_wcr = c.dec

# initial conditions Apep
# all initial conditions are can be found in the following article: https://watermark.silverchair.com/staa3863.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3wwggN4BgkqhkiG9w0BBwagggNpMIIDZQIBADCCA14GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMKndlGG2Ra9RkEdg-AgEQgIIDL6wE--nTkGZ8e0rh88eTXuBAWY7b3ITD86GHidYHDCGBOro-ASIs5w35zYRTYkhoAi8u5DNEE6LZeqHiCKKTLYjh_p01AaI7Zi12PX8h4PZc_K37njEouZsAy7YfKQIurbW-NDPGSVHShgMfgHwD5FryxSTw9PXeMoBqCRZsYW2SVBTHdkQKyzNEFEp29S97KzzD8ZDEpJulpEKZ8K63mm2ZIDt0HrwqaI0GEtrX6ps1t3uHEqexJj2ashk_SlvLN_D6UtRCnNBqLi5CBl9jw0nBWBiU3eNJTex2BddwEuGVlTeTehBelYquj0w-RaGfLXK5Q2vd0BaF2B04G8tYLyHjZpEg9VoaKyh33eHrts77el-s4ArXDvRKeMVa5-GiKiVdG24NWyWe5sMHVphEsAqOjyfu8YNldFWpRgFUr21umK3f3QoKlZDFw3SWqR-mvuT_d3AlkGUKpobIkA5HcK9_8HVy8bjRTGar9EQbjDLerOdfKB7W6fm2L3AY2o_cpHKbbgdy8-6hC9KOeZzLh3pYp9TWhFqp7IDHK7SczLEK0S5mXIjVm2VnxhXxt3Z7XEhPmLGMKPNPlUYpCxUeGXbjdylAkv5fNknQ_DifS606RN_znDskUOx9v5YWZALBoSdhGHbymbyuXvjokrJgagtDFVmXeLcgifPbkkePDLNGIoGewQA5LquVfqkaMyKVbfMCAT0sAlZb0FRt3r_PhVFZO-rh06dumI875X3NaERyoCg8SY5RCncwXLlLoBjNl-b9YkowdwBKnfYn6FUMK4oQ4GGMK6LJa_Lnu9azyn8yIOb1DGX4RQRnVxeRsualYVuqyNJqzrIA1XSyNYbYp0BYQJWDPfACeegVweEtfrbAWCpDsWq5WOo4QpUFiPMMmtuo0gD27YX35EExxoKyWCMfCFQ1K0169CLf_-My_MjQ_iU0PSzMWpbbpIRgYA0jKz9HeKNps1iTMec7gzaP-eDa36x_32lpW5jZfm_W6zBbOt5v7FZ5Jy6jgmq6NgDaGN-kskZhGZal3OikElV0-qx1XIOS8ei-CVAIL2doUFcAFF07d-5inXsoikaF8bom
#M_0_O = 10**-4.5 * M_s  
#M_0_wr = 10**-4.3 * M_s  
#v_O = 2100 
#v_wr = 3500 
#D_AU = 112.3 
#D_mas = 47 * u.mas
#alpha = (360-277+90) * u.deg 
#c = SkyCoord('16h00m50.4846697s','-51d42m45.3384857s') 
#alpha_wcr = c.ra 
#delta_wcr= c.dec 

# all calculations are based on Canto et al (1996)
# calculations within the coordinate system of the O-type star
# calculation mass loss for both stars
# O-type star
def massloss_O(theta): 
    return M_0_O/2 * (1-np.cos(theta))

# WR-type star
def massloss_wr(theta): 
    return M_0_wr/2 * (1-np.cos(theta))

# calculating beta constant (momentum ratio)
b = (M_0_O*v_O)/(M_0_wr*v_wr) 

# calculating theta WR 
def theta_wr(theta):
    if isinstance(theta,u.Quantity):
        th = theta.to(u.rad).value
        theta1 = (15/2*(-1+np.sqrt(1+4/5*b*(1-th*(1/np.tan(th))))))**(1/2) 
        return (theta1 * u.rad).to(theta.unit)
    else:
        theta1 = (15/2*(-1+np.sqrt(1+4/5*b*(1-theta*(1/np.tan(theta))))))**(1/2) 
    return theta1

# numerical approach by Benito Marcote based on Canto et al
def get_theta1(theta, eta):
        """Determines theta1 given eta and theta.
        If theta does not have units, radians are assumed.
        """
        if isinstance(theta, u.Quantity):
            f = lambda theta1 : np.nan_to_num(theta1/np.tan(theta1), nan=1.0) - 1 - \
                eta*(np.nan_to_num(theta.to(u.rad).value/np.tan(theta), nan=1.0) - 1)
            return (fsolve(f,  theta.to(u.rad).value)*u.rad).to(theta.unit)
        else:
            f = lambda theta1 : np.nan_to_num(theta1/np.tan(theta1), nan=1.0) - 1 - \
            eta*(np.nan_to_num(theta/np.tan(theta), nan=1.0) - 1)
            return fsolve(f, theta)

# calculating the radius
def R(theta_O,theta_wr,D): 
    radius = D*np.nan_to_num(np.sin(theta_wr)*(1/np.sin(theta_O+theta_wr)))
    return radius

# lists to store information
xcoords = []
ycoords = []
xcoords_under = []
ycoords_under = []

# calculating x and y coordinates of the shockfront
for i in np.arange(0.01,np.pi/2,0.01):
    x = R(i,theta_wr(i),D_AU) * np.cos(i) #au
    y = R(i,theta_wr(i),D_AU) * np.sin(i) #au
    xcoords.append(x)
    ycoords.append(y)

# coordinates of the two stars
# O-type star
O_x = 0
O_y = 0

# WR-type star
wr_x = D_AU
wr_y = 0

# make graph symetrical below the x-axis
for x in xcoords:
    xcoords_under.append(x)

for y in ycoords:
    y = y * -1
    ycoords_under.append(y)

# stagnation point in AU
R0 = (b**(1/2)*D_AU)/(1+b**(1/2)) 

# plot stagnation point
R0_xcoords = [0,R0]
R0_ycoords = [0,0]

# plot graph
plt.plot(xcoords,ycoords, color = 'lightskyblue', label = 'shockfront')
plt.plot(xcoords_under,ycoords_under, color = 'lightskyblue')
#plt.scatter(O_x,O_y, linewidths=10, label = 'O-type star', color = 'plum')
#plt.scatter(wr_x,wr_y, linewidths=10, label = 'WR-type star', color = 'lime')
plt.scatter(O_x,O_y, linewidths=10, label = 'WC-type star', color = 'plum')
plt.scatter(wr_x,wr_y, linewidths=10, label = 'WN-type star', color = 'lime')
plt.plot(R0_xcoords,R0_ycoords, color ='g', label = 'R0', linestyle = '--')
plt.xlabel("distance (AU)")
plt.ylabel("distance (AU)")
plt.legend()
#plt.savefig('C:/Users/roosu/OneDrive/bachelor project/shockfront (from within binary) WR140.png')
#plt.savefig('C:/Users/roosu/OneDrive/bachelor project/shockfront (from within binary) Apep.png')
#plt.show()

# placing the binary system in the sky 
# coordinate system as if observed from earth
# coordinate system change is based on Marcote et al (2020)
# calculate skycoords
def skycoords(thetas, alpha, alpha_wcr, delta_wcr, distance, b, y_is_declination=False):
    # get theta WR
    theta_wrs = get_theta1(thetas,b)
    #theta_wrs = theta_wr(thetas)

    # coordinate of contact discontinuity
    x_cd = R(thetas, theta_wrs, distance) * np.cos(thetas) 
    y_cd = R(thetas, theta_wrs, distance) * np.sin(thetas)
    
    # stagnation point
    R_O = (np.sqrt(b) * distance) / (1 + np.sqrt(b))

    # coordinate system change
    x2 = ((x_cd - R_O) * np.cos(alpha)) - (y_cd * np.sin(alpha)) 
    y2 = ((x_cd - R_O) * np.sin(alpha)) + (y_cd * np.cos(alpha)) 

    if y_is_declination:
        alpha_cd = x2 / np.cos(delta_wcr) + alpha_wcr 
        delta_cd = y2 + delta_wcr 
        return alpha_cd, delta_cd

    alpha_cd = x2 + alpha_wcr 
    delta_cd = y2 + delta_wcr    
    return alpha_cd, delta_cd

thetas = np.linspace(-80,96,55)
coords_alpha_cd, coords_delta_cd = skycoords(thetas[np.where(thetas != 0.0)]*u.deg, alpha, alpha_wcr, delta_wcr, D_mas.to(u.deg), b, y_is_declination=True)

# plot the two stars
# using the stagnation point and coordinate change 
# O-type star
R0 = (np.sqrt(b) * D_mas.to(u.deg)) / (1 + np.sqrt(b))
x_O = ((-R0 * np.cos(alpha))/np.cos(delta_wcr)) + alpha_wcr
y_O = (-R0 * np.sin(alpha)) + delta_wcr
print(f"alpha = {x_O.to(u.hourangle).to_string(sep='hms')}")
print(f"delta = {y_O.to(u.deg).to_string(sep='dms')} ")

# WR-type star
Rwr = D_mas.to(u.deg)/(1 + np.sqrt(b))
x_wr = (Rwr * np.cos(alpha))/np.cos(delta_wcr) + alpha_wcr
y_wr = Rwr * np.sin(alpha) + delta_wcr
print(f"alpha = {x_wr.to(u.hourangle).to_string(sep='hms')}")
print(f"delta = {y_wr.to(u.deg).to_string(sep='dms')} ")

# creating format to plot axis in hourangles and degrees
def ra_formatter(x, pos):
    return Angle(x * u.deg).to_string(unit=u.hourangle, sep=('h','m','s'), pad=True)

def dec_formatter(x, pos):
    return Angle(x * u.deg).to_string(unit=u.deg, sep=('d','m','s'), alwayssign=True, pad=True)

# plot graph
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(coords_alpha_cd, coords_delta_cd, '-', lw = 4, label = 'shock front', color = 'lightskyblue')
#ax.scatter(x_O, y_O, label = 'O-type star', color = 'plum')
#ax.scatter(x_wr, y_wr, label = 'wr-type star', color = 'lime')
ax.scatter(x_O, y_O, label = 'WC-type star', color = 'plum')
ax.scatter(x_wr, y_wr, label = 'WN-type star', color = 'lime')
ax.xaxis.set_major_formatter(FuncFormatter(ra_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(dec_formatter))
ax.xaxis.set_tick_params(labelsize = 6)
ax.yaxis.set_tick_params(labelsize = 7)
ax.set_xlabel("Right Ascension (J2000, h:m:s)")
ax.set_ylabel("Declination (J2000, °:′:″)")
ax.invert_xaxis()     
ax.legend()
fig.tight_layout(pad=0.5)
#fig.savefig('C:/Users/roosu/OneDrive/bachelor project/shockfront (as observed from earth) WR140.png')
#fig.savefig('C:/Users/roosu/OneDrive/bachelor project/shockfront (as observed from earth) Apep.png')
#plt.show()

# calculate error margins
# Apep
# error on apep: RA  +- 0.3 mas           https://watermark.silverchair.com/staa3863.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA1MwggNPBgkqhkiG9w0BBwagggNAMIIDPAIBADCCAzUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMNwdwA9bE-fey03-fAgEQgIIDBtE2zpZjTg4bd3lPNujG8RDCI4szKM1THMcECVH8yiEZ0AEfuGjJKWVhvMGyuFdfLsvvl2qCNSJ5X5JUEqUYIjnY5TYY2_MiqDEq88pBUqoUtj0u7dZT9FhZViVvG4srhyrhpq7p41aCK-VkhJVOobCzAzukvF-ZwjLhPEG_-hRQrusNEayizaW3J2TExDTxpJOG7YCWvQ0pS3SkHpwNt2fAnPnpli-6bHhYv2tP-iUfME5vxJZ_yIcH_rdhClmW0It6iTFSSZ9SiBjPJacyiOsF3xUlzF9PBn7LIMwOk-NfckFCCl33Q68x4C3t0GZKxVOTKmzts3Zyxnr8Kx4V-DfsXgKQovx8zFT04Vnfk4nRKvqNHb7PkRTRgaUPyXmiubNNoyzOjvFanP70HhYKYQS5__vTcO9F5-Ys_a5RjXdRrtFxZaQYbVx901bPWkCmI-IIjtfalwCYLU-s-x0qteBBD-DS0mkxpCm6CGHTt-roL4f1JYB_qaVQvS0dyHraeadFtHrFy4XtradyCBYeCHgC4cQwN9ToJDn3V7ETCmWKHe_4tkBrSkBmes0_vpQKZflISSDKMXFOqszKE8RAc7zZNSjmu6f61k2eWqkMLroz9xKJHmDWmpmdsbpH5NZQ0urJLha6CVd-KU3NOlQLf6FJIOCorExtrneHADVfKYMKAVY8sQ1gdVlnhYD08BF_GCWcwpNgmiqw_JjuW-wXlo3Zsv8pkzbR85vbyn2-c32zElPGd6v4wQvhGi1qsQgEE3JdriMZQTgDRQYvIH51TxYOOU-FNxXjRQeYLYwNeWnjMdeM9uD4rbHXZTy3yUnboYtUvBFtDAh5M7xW596ElrEZXs19dSPDkT0YtVNDSwsqGLQB3RG6La_aFSXOyqw25YGYhfgGPh7akuJFTfnzs6MlQGlO-8VOg0ojyLHnxuXd_wCk0As9gt6i1BpMH_hi35EkchIyBPe4dtuTqVr_Z0nJOqR4ONgZjH7LqBr4vf9Ck37gk_CzNDdpOF4HUWh2w28BuQitfg
#                Dec +- 0.5 mas           https://watermark.silverchair.com/staa3863.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA1MwggNPBgkqhkiG9w0BBwagggNAMIIDPAIBADCCAzUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMNwdwA9bE-fey03-fAgEQgIIDBtE2zpZjTg4bd3lPNujG8RDCI4szKM1THMcECVH8yiEZ0AEfuGjJKWVhvMGyuFdfLsvvl2qCNSJ5X5JUEqUYIjnY5TYY2_MiqDEq88pBUqoUtj0u7dZT9FhZViVvG4srhyrhpq7p41aCK-VkhJVOobCzAzukvF-ZwjLhPEG_-hRQrusNEayizaW3J2TExDTxpJOG7YCWvQ0pS3SkHpwNt2fAnPnpli-6bHhYv2tP-iUfME5vxJZ_yIcH_rdhClmW0It6iTFSSZ9SiBjPJacyiOsF3xUlzF9PBn7LIMwOk-NfckFCCl33Q68x4C3t0GZKxVOTKmzts3Zyxnr8Kx4V-DfsXgKQovx8zFT04Vnfk4nRKvqNHb7PkRTRgaUPyXmiubNNoyzOjvFanP70HhYKYQS5__vTcO9F5-Ys_a5RjXdRrtFxZaQYbVx901bPWkCmI-IIjtfalwCYLU-s-x0qteBBD-DS0mkxpCm6CGHTt-roL4f1JYB_qaVQvS0dyHraeadFtHrFy4XtradyCBYeCHgC4cQwN9ToJDn3V7ETCmWKHe_4tkBrSkBmes0_vpQKZflISSDKMXFOqszKE8RAc7zZNSjmu6f61k2eWqkMLroz9xKJHmDWmpmdsbpH5NZQ0urJLha6CVd-KU3NOlQLf6FJIOCorExtrneHADVfKYMKAVY8sQ1gdVlnhYD08BF_GCWcwpNgmiqw_JjuW-wXlo3Zsv8pkzbR85vbyn2-c32zElPGd6v4wQvhGi1qsQgEE3JdriMZQTgDRQYvIH51TxYOOU-FNxXjRQeYLYwNeWnjMdeM9uD4rbHXZTy3yUnboYtUvBFtDAh5M7xW596ElrEZXs19dSPDkT0YtVNDSwsqGLQB3RG6La_aFSXOyqw25YGYhfgGPh7akuJFTfnzs6MlQGlO-8VOg0ojyLHnxuXd_wCk0As9gt6i1BpMH_hi35EkchIyBPe4dtuTqVr_Z0nJOqR4ONgZjH7LqBr4vf9Ck37gk_CzNDdpOF4HUWh2w28BuQitfg
# error on beta:     +- 0.08 (no unit)    https://watermark.silverchair.com/staa3863.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA1MwggNPBgkqhkiG9w0BBwagggNAMIIDPAIBADCCAzUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMNwdwA9bE-fey03-fAgEQgIIDBtE2zpZjTg4bd3lPNujG8RDCI4szKM1THMcECVH8yiEZ0AEfuGjJKWVhvMGyuFdfLsvvl2qCNSJ5X5JUEqUYIjnY5TYY2_MiqDEq88pBUqoUtj0u7dZT9FhZViVvG4srhyrhpq7p41aCK-VkhJVOobCzAzukvF-ZwjLhPEG_-hRQrusNEayizaW3J2TExDTxpJOG7YCWvQ0pS3SkHpwNt2fAnPnpli-6bHhYv2tP-iUfME5vxJZ_yIcH_rdhClmW0It6iTFSSZ9SiBjPJacyiOsF3xUlzF9PBn7LIMwOk-NfckFCCl33Q68x4C3t0GZKxVOTKmzts3Zyxnr8Kx4V-DfsXgKQovx8zFT04Vnfk4nRKvqNHb7PkRTRgaUPyXmiubNNoyzOjvFanP70HhYKYQS5__vTcO9F5-Ys_a5RjXdRrtFxZaQYbVx901bPWkCmI-IIjtfalwCYLU-s-x0qteBBD-DS0mkxpCm6CGHTt-roL4f1JYB_qaVQvS0dyHraeadFtHrFy4XtradyCBYeCHgC4cQwN9ToJDn3V7ETCmWKHe_4tkBrSkBmes0_vpQKZflISSDKMXFOqszKE8RAc7zZNSjmu6f61k2eWqkMLroz9xKJHmDWmpmdsbpH5NZQ0urJLha6CVd-KU3NOlQLf6FJIOCorExtrneHADVfKYMKAVY8sQ1gdVlnhYD08BF_GCWcwpNgmiqw_JjuW-wXlo3Zsv8pkzbR85vbyn2-c32zElPGd6v4wQvhGi1qsQgEE3JdriMZQTgDRQYvIH51TxYOOU-FNxXjRQeYLYwNeWnjMdeM9uD4rbHXZTy3yUnboYtUvBFtDAh5M7xW596ElrEZXs19dSPDkT0YtVNDSwsqGLQB3RG6La_aFSXOyqw25YGYhfgGPh7akuJFTfnzs6MlQGlO-8VOg0ojyLHnxuXd_wCk0As9gt6i1BpMH_hi35EkchIyBPe4dtuTqVr_Z0nJOqR4ONgZjH7LqBr4vf9Ck37gk_CzNDdpOF4HUWh2w28BuQitfg
# WR 140
# error on WR 140: RA  +- 18.6 mas      https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=WR140&submit=SIMBAD+search
#                  Dec +- 21.2 mas      https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=WR140&submit=SIMBAD+search
# error on beta:       +- cannot find this...

error_RA = 18.6 * u.mas
error_RA_HA = error_RA.to(u.deg)
error_dec =  21.2 * u.mas  
error_dec_deg = error_dec.to(u.deg)
error_beta = 0.00

list_alpha_O_error = []
list_delta_O_error = []
list_alpha_wr_error = []
list_delta_wr_error = []

# 1: RA=+0.3 Dec=+0.5 beta=+0.08
b_pos = b + error_beta
R01 = (np.sqrt(b_pos) * D_mas.to(u.deg)) / (1 + np.sqrt(b_pos))
alpha_wcrpos = alpha_wcr + error_RA_HA
alpha_wcrneg = alpha_wcr - error_RA_HA
delta_wcrpos = delta_wcr + error_dec_deg
delta_wcrneg = delta_wcr -error_dec_deg
Rwr1 = D_mas.to(u.deg)/(1 + np.sqrt(b_pos))

alpha_O_1 = ((-R01 * np.cos(alpha))/np.cos(delta_wcrpos)) + alpha_wcrpos
delta_O_1 = (-R01 * np.sin(alpha)) + delta_wcrpos
alpha_wr_1 = (Rwr1 * np.cos(alpha))/np.cos(delta_wcrpos) + alpha_wcrpos
delta_wr_1 = Rwr1 * np.sin(alpha) + delta_wcrpos
list_alpha_O_error.append(alpha_O_1)
list_delta_O_error.append(delta_O_1)
list_alpha_wr_error.append(alpha_wr_1)
list_delta_wr_error.append(delta_wr_1)
#print("1:", alpha_O_1, delta_O_1, alpha_wr_1, delta_O_1)

# 2: RA=-0.3 Dec=+0.5 beta=+0.08
alpha_O_2 = ((-R01 * np.cos(alpha))/np.cos(delta_wcrpos)) + alpha_wcrneg
delta_O_2 = (-R01 * np.sin(alpha)) + delta_wcrpos
alpha_wr_2 = (Rwr1 * np.cos(alpha))/np.cos(delta_wcrpos) + alpha_wcrneg
delta_wr_2 = Rwr1 * np.sin(alpha) + delta_wcrpos
list_alpha_O_error.append(alpha_O_2)
list_delta_O_error.append(delta_O_2)
list_alpha_wr_error.append(alpha_wr_2)
list_delta_wr_error.append(delta_wr_2)
#print("2:", alpha_O_2, delta_O_2, alpha_wr_2, delta_O_2)

# 3: RA=-0.3 Dec=-0.5 beta=+0.08
alpha_O_3 = ((-R01 * np.cos(alpha))/np.cos(delta_wcrneg)) + alpha_wcrneg
delta_O_3 = (-R01 * np.sin(alpha)) + delta_wcrneg
alpha_wr_3 = (Rwr1 * np.cos(alpha))/np.cos(delta_wcrneg) + alpha_wcrneg
delta_wr_3 = Rwr1 * np.sin(alpha) + delta_wcrneg
list_alpha_O_error.append(alpha_O_3)
list_delta_O_error.append(delta_O_3)
list_alpha_wr_error.append(alpha_wr_3)
list_delta_wr_error.append(delta_wr_3)
#print("3:", alpha_O_3, delta_O_3, alpha_wr_3, delta_O_3)

# 4: RA=0.3 Dec=-0.5 beta=+0.08
alpha_O_4 = ((-R01 * np.cos(alpha))/np.cos(delta_wcrneg)) + alpha_wcrpos
delta_O_4 = (-R01 * np.sin(alpha)) + delta_wcrneg
alpha_wr_4 = (Rwr1 * np.cos(alpha))/np.cos(delta_wcrneg) + alpha_wcrpos
delta_wr_4 = Rwr1 * np.sin(alpha) + delta_wcrneg
list_alpha_O_error.append(alpha_O_4)
list_delta_O_error.append(delta_O_4)
list_alpha_wr_error.append(alpha_wr_4)
list_delta_wr_error.append(delta_wr_4)
#print("4:", alpha_O_4, delta_O_4, alpha_wr_4, delta_O_4)

# 5: RA=0.3 Dec=0.5 beta=-0.08
b_neg = b - error_beta
R02 = (np.sqrt(b_neg) * D_mas.to(u.deg)) / (1 + np.sqrt(b_neg))
Rwr2 = D_mas.to(u.deg)/(1 + np.sqrt(b_neg))
alpha_O_5 = ((-R02 * np.cos(alpha))/np.cos(delta_wcrpos)) + alpha_wcrpos
delta_O_5 = (-R02 * np.sin(alpha)) + delta_wcrpos
alpha_wr_5 = (Rwr2 * np.cos(alpha))/np.cos(delta_wcrpos) + alpha_wcrpos
delta_wr_5 = Rwr2 * np.sin(alpha) + delta_wcrpos
list_alpha_O_error.append(alpha_O_5)
list_delta_O_error.append(delta_O_5)
list_alpha_wr_error.append(alpha_wr_5)
list_delta_wr_error.append(delta_wr_5)
#print("5:", alpha_O_5, delta_O_5, alpha_wr_5, delta_O_5)

# 6: RA=-0.3 Dec=0.5 beta=-0.08
alpha_O_6 = ((-R02 * np.cos(alpha))/np.cos(delta_wcrpos)) + alpha_wcrneg
delta_O_6 = (-R02 * np.sin(alpha)) + delta_wcrpos
alpha_wr_6 = (Rwr2 * np.cos(alpha))/np.cos(delta_wcrpos) + alpha_wcrneg
delta_wr_6 = Rwr2 * np.sin(alpha) + delta_wcrpos
list_alpha_O_error.append(alpha_O_6)
list_delta_O_error.append(delta_O_6)
list_alpha_wr_error.append(alpha_wr_6)
list_delta_wr_error.append(delta_wr_6)
#print("6:", alpha_O_6, delta_O_6, alpha_wr_6, delta_O_6)

# 7: RA=-0.3 Dec=-0.5 beta=-0.08
alpha_O_7 = ((-R02 * np.cos(alpha))/np.cos(delta_wcrneg)) + alpha_wcrneg
delta_O_7 = (-R02 * np.sin(alpha)) + delta_wcrneg
alpha_wr_7 = (Rwr2 * np.cos(alpha))/np.cos(delta_wcrneg) + alpha_wcrneg
delta_wr_7 = Rwr2 * np.sin(alpha) + delta_wcrneg
list_alpha_O_error.append(alpha_O_7)
list_delta_O_error.append(delta_O_7)
list_alpha_wr_error.append(alpha_wr_7)
list_delta_wr_error.append(delta_wr_7)
#print("7:", alpha_O_7, delta_O_7, alpha_wr_7, delta_O_7)

# 8: RA=0.3 Dec=-0.5 beta=-0.08
alpha_O_8 = ((-R02 * np.cos(alpha))/np.cos(delta_wcrneg)) + alpha_wcrpos
delta_O_8 = (-R02 * np.sin(alpha)) + delta_wcrneg
alpha_wr_8 = (Rwr2 * np.cos(alpha))/np.cos(delta_wcrneg) + alpha_wcrpos
delta_wr_8 = Rwr2 * np.sin(alpha) + delta_wcrneg
list_alpha_O_error.append(alpha_O_8)
list_delta_O_error.append(delta_O_8)
list_alpha_wr_error.append(alpha_wr_8)
list_delta_wr_error.append(delta_wr_8)
#print("8:", alpha_O_8, delta_O_8, alpha_wr_8, delta_O_8)

# find max and min
alpha_O_max = max(list_alpha_O_error)
alpha_O_min = min(list_alpha_O_error)
delta_O_max = max(list_delta_O_error)
delta_O_min = min(list_delta_O_error)
alpha_wr_max = max(list_alpha_wr_error)
alpha_wr_min = min(list_alpha_wr_error)
delta_wr_max = max(list_delta_wr_error)
delta_wr_min = min(list_delta_wr_error)

# second idea to calculate
final_error_alpha_O1 = (alpha_O_max - x_O).to(u.hourangle).to_string(sep='hms')
final_error_alpha_O2 = (alpha_O_min - x_O).to(u.hourangle).to_string(sep='hms')
final_error_delta_O1 = (delta_O_max - y_O).to(u.deg).to_string(sep='dms')
final_error_delta_O2 = (delta_O_min - y_O).to(u.deg).to_string(sep='dms')
final_error_alpha_wr1 = (alpha_wr_max - x_wr).to(u.hourangle).to_string(sep='hms')
final_error_alpha_wr2 = (alpha_wr_min - x_wr).to(u.hourangle).to_string(sep='hms')
final_error_delta_wr1 = (delta_wr_max - y_wr).to(u.deg).to_string(sep='dms')
final_error_delta_wr2 = (delta_wr_min - y_wr).to(u.deg).to_string(sep='dms')

print(final_error_alpha_O1, final_error_alpha_O2)
print(final_error_delta_O1, final_error_delta_O2)
print(final_error_alpha_wr1, final_error_alpha_wr2)
print(final_error_delta_wr1, final_error_alpha_wr2)
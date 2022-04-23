import numpy as np
import fluids.atmosphere as atm

Re = 6378000
g0 = atm.ATMOSPHERE_1976(0).gravity(0)
w = 0
pi = np.pi

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def tan(x):
    return np.tan(x)

# Funcoes para a solucao das EDOS
def drag(rho, v, A_drag):
    return 0.5 * rho * v ** 2 * A_drag

def dvdt(T, D, m, gam, phi, delta, x, eps, mu):
    T = T*cos(eps)*cos(mu)
    return (T - D) / m - g0 * sin(gam) + w**2*(Re + x)*cos(delta)**2*(sin(gam) - cos(gam)*tan(delta)*sin(phi))

def dphidt(T, m, v, x, gam, phi, delta, mu):
    T = T*sin(mu)
    return T/(m*cos(gam)) - cos(gam) * v * cos(phi) * tan(delta)/(Re+x) + 2*w*(tan(gam)*cos(delta)*sin(phi)-sin(delta)) - w**2*(Re+x)*sin(delta)*cos(delta)*cos(phi)/cos(gam)

def dgam_dt(T, m, v, x, gam, phi, delta, eps, mu):
    T = T*sin(eps)*cos(mu)
    return T/(m*v) - ((g0 / v) - v / (Re + x)) * cos(gam) + 2*w*v*cos(delta)*cos(phi) + w**2*(Re+x)*cos(delta)**2*(cos(gam) + sin(gam)*tan(delta)*sin(phi))

def ddelta_dt(v, gam, phi, x):
    return v*cos(gam)*sin(phi)/(Re+x)

def dydt(v, gam, phi, uwind):
    return v*cos(gam)*cos(phi) - uwind

def dxdt(v, gam, phi, vwind):
    return v*cos(gam)*sin(phi) - vwind

def dzdt(v, gam):
    return v * sin(gam)

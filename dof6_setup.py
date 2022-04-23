# Atmospheric dada import
import fluids.atmosphere as atm

# Utils bibs and functions imported
import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
from dof3_functions import *

# Wind bibs and functions needed are imported
import netCDF4 as nc
from wind_model.mle import *

# Graphical results library
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Interpolação dos dados aerodinâmicos empíricos, utilizados para o arrasto com alfa>0
x_cd = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20])
y_eta = np.array([0.6, 0.6193, 0.6605, 0.6835, 0.7107, 0.7263, 0.7421, 0.7535, 0.7636])
y_del = np.array([0.7802, 0.8598, 0.9196, 0.9396, 0.96, 0.97, 0.9764, 0.9817, 0.9856])
cs_eta = CubicSpline(x_cd, y_eta)
cs_del = CubicSpline(x_cd, y_del)


# Funções utilizadas frequentemente
def cos(a):
    return np.cos(a)

def sen(a):
    return np.sin(a)

def tan(a):
    return np.tan(a)

# Para os propósitos de cáculos, é assumido que o veículo é axissimétrico
# e o CM dele e suas componentes pertencem ao eixo de simetria (x no ref do corpo)
# (Estima-se o parâmetros dinâmicos e aerodinâmicos desse veículo passivamente controlado)
class simulation():
    def __init__(self, **kwargs):
        self.drogue_chute       = kwargs.get('add_drogue')
        self.main_chute         = kwargs.get('add_main')
        self.motor_missalign    = kwargs.get('add_miss')
        self.step               = kwargs.get('step_time')
        self.tol                = kwargs.get('tolerance')
        self.wind               = kwargs.get('add_wind')
        self.plot_trajectory    = kwargs.get('plot_3dtraj')
        self.plot_t_trajectory  = kwargs.get('plot_result')
        self.save               = kwargs.get('save_result')
        self.save_name          = kwargs.get('save_name')

    def motormissalign(self, **kwargs):
        if self.motor_missalign:
            self.eps = np.deg2rad(kwargs.get('eps'))
            self.mu  = np.deg2rad(kwargs.get('mu'))
        else:
            self.eps = 0
            self.mu  = 0

    def droguechute(self, **kwargs):
        if self.drogue_chute:
            self.Cd_drogue          = kwargs.get('Cd')
            self.diameter_drogue    = kwargs.get('diameter')
            self.time_open          = kwargs.get('open_time')
        else:
            self.Cd_drogue          = 0
            self.diameter_drogue    = 0
            self.time_open          = 0

    def mainchute(self, **kwargs):
        if self.main_chute:
            self.Cd_main            = kwargs.get('Cd')
            self.diameter_main      = kwargs.get('diameter')
            self.h_open             = kwargs.get('altitude')
        else:
            self.Cd_main            = 0
            self.diameter_main      = 0
            self.h_open             = 0

class rocket():
    def __init__(self):
        pass

    def initial(self, **initial_cond):
        # Algumas constantes
        self.wt = 7.27E-5 # Velocidade angular da terra
        self.Rt = 6.37814E3 # Raio da terra [m]
        self.g0 = atm.ATMOSPHERE_1976(0).g

        # Convertion factos
        deg2rad = np.pi/180
        # Condições iniciais
        self.l_trilho   = initial_cond.get('rail_length')
        self.latitude   = np.deg2rad(float(initial_cond.get('latitude')))
        self.longitude  = np.deg2rad(float(initial_cond.get('longitude')))
        self.azimute    = np.deg2rad(float(initial_cond.get('azimut')))
        self.elevation  = np.deg2rad(float(initial_cond.get('elevation')))
        self.phi0       = np.deg2rad(float(initial_cond.get('phi0')))
        self.u0         = initial_cond.get('u0')
        self.v0         = initial_cond.get('v0')
        self.w0         = initial_cond.get('w0')
        self.Va         = initial_cond.get('Va')
        self.p0         = initial_cond.get('p0')
        self.q0         = initial_cond.get('q0')
        self.r0         = initial_cond.get('r0')
        self.x0         = initial_cond.get('x0')
        self.y0         = initial_cond.get('y0')
        self.z0         = initial_cond.get('z0')
        self.alpha0     = np.deg2rad(float(initial_cond.get('alpha')))
        self.beta0      = np.deg2rad(float(initial_cond.get('beta')))
        self.delta      = np.deg2rad(float(initial_cond.get('delta')))
        self.Mi         = initial_cond.get('init_mass')

    def fuselage(self, **kwargs):
        self.FusLength  = kwargs.get('Fus_length')                                                   # Comprimento da fuselagem
        self.FusRho     = kwargs.get('Fus_dens')                                                      # Fuselage material Density
        self.de         = kwargs.get('ex_diameter')                                                         # Diâmetro externo
        self.di         = kwargs.get('in_diameter')                                     # Diâmetro interno
        self.db         = kwargs.get('bs_diameter')                                                         # Diâmetro da base (caso boattail)
        self.FusTick    = (self.de-self.di)/2
        self.FusMass    = self.FusRho*np.pi*((self.de/2)**2 - (self.de/2 - self.FusTick)**2)*self.FusLength # Fuselage mass
        self.Sbt        = 0.25*np.pi*self.de**2                                        # Reference area (maximum external fuselage diameter)
        self.Swt        = 2*np.pi*self.de*self.FusLength                               # Área do tubo
        self.Swtt       = self.de*self.FusLength                                      # Área molhada da fuselagem

    def nosecone(self, **kwargs):
        # Propriedades do NoseCone
        self.NoseRho    = kwargs.get('Nose_dens')                                                     # Densidade do nose (Kg/m**3)
        self.NoseLength = kwargs.get('Nose_length')                                                  # Comprimento do nose [m]
        self.NoseMass   = kwargs.get('Nose_mass')
        self.NoseRad    = self.de/2                                                # Nose cone raio da base [m]
        eps_nose        = np.sqrt(self.NoseLength**2 - (self.de/2)**2)/self.NoseLength
        self.Sw_bt_nose = np.pi*self.NoseLength**2 + 0.5*np.pi*np.log((1+eps_nose)/(1-eps_nose))*(self.de/2)**2                                         # Área molhada do NOSE
        self.f          = self.NoseLength/self.de                                        # Fineness L/2r

    def fins(self, **kwargs):
        # Propriedades das aletas
        self.N          = kwargs.get('Number')                                                              # Número de aletas
        self.Ct         = kwargs.get('Tip_chord')                                                          # Corda no topo
        self.Cr         = kwargs.get('Root_chord')                                                          # Corda na raíz
        self.b          = kwargs.get('semi_span')                                                          # (span)
        self.tick       = kwargs.get('max_tick')
        self.sweep_angle= np.deg2rad(kwargs.get('sweep'))                                                      # Espessura
        # self.X_tc_av    = 0.3
        self.S          = (self.Ct+self.Cr)*self.b/2                                     # Área planar
        self.AR         = self.b**2/self.S                                                   # Aspect Ratio
        self.lam        = self.Ct/self.Cr                                              # Taper Ratio
        self.tick_c     = self.tick/self.Cr                                               # Espessura pela corda
        self.lw         = self.Cr - self.Ct

        # Propriedades geometricas globais
        self.L_total    = self.NoseLength + self.FusLength                         # Comprimento total do foguete
        self.Sw_bt      = self.Sw_bt_nose + self.Swtt                                # Área molhada total sem aletas
        self.S_total    = self.Sw_bt + self.S*self.N                               # Área molhada total do foguete
        self.Xf         = self.L_total - self.Cr                                        # Distância entre a ponta do nose e o início da aleta

    def propulsion(self, **kwargs):
        file_name       = kwargs.get('file_name')
        type            = kwargs.get('type')
        delimiter       = kwargs.get('delimiter')
        prop_data       = np.genfromtxt(file_name, delimiter=delimiter)
        prop_data[0][0] = 0.0
        if type.lower() == 'solid':
            self.t      = prop_data[:,0]                                                        # Tempo de empuxo
            Mdot        = prop_data[:,1]                                                      # Fluxo mássico de combustível
            self.T      = prop_data[:,2]                                                         # Empuxo
            self.M      = np.array([self.Mi])
            k           = self.t[1]-self.t[0]
            for i in range(1, len(self.t)):
                self.M  = np.append(self.M, self.M[i-1] - M_dot[i]*k)                                       # Massa do foguete como f(t)
            M_final     = self.M[-1]

        if type.lower() == 'hybrid':
            self.t      = prop_data[:,0]                                                        # Tempo de empuxo
            Mdot_fu     = prop_data[:,1]                                                      # Fluxo mássico de combustível
            Mdot_ox     = prop_data[:,2]                                                      # Fluxo mássico de combustível
            self.T      = prop_data[:,3]                                                         # Empuxo
            M_dot       = Mdot_fu + Mdot_ox
            self.M      = np.array([self.Mi])
            k           = self.t[1]-self.t[0]
            for i in range(1, len(self.t)):
                self.M  = np.append(self.M, self.M[i-1] - M_dot[i]*k)                                       # Massa do foguete como f(t)
            M_final     = self.M[-1]

    def structural(self, **kwargs):
        file_name       = kwargs.get('file_name')
        delimiter       = kwargs.get('delimiter')
        struc_data      = np.genfromtxt(file_name, delimiter=delimiter)
        self.Xcm        = struc_data[:,0]
        self.Ixx        = struc_data[:,1]
        self.Iyy        = struc_data[:,2]
        self.Izz        = struc_data[:,3]

class aerodynamic():
    # Cálculos aerodinâmicos com base em 'Estimating the dynamic and aerodynamic paramters of
    # passively controlled high power rockets for flight simulaton'

    def __init__(self, launch_vehicle, v=0.001, rho = 1.250, M = 0.0001, Xcm = 0, p = 0, q = 0, r = 0, alfa = 0, beta = 0, delta = 0):
        # Normal alpha coefficient and Center of Pressure
        # print('Aero_input: ', rho, v, M, Xcm)
        Cd = self.axial(alfa, v, rho, M, launch_vehicle)

        if Xcm == 0 and p == 0 and q == 0 and r == 0:
            self.forcecoeff = np.array([0, Cd, 0, 0, 0, 0, 0], dtype=object)
        else:
            Xcp, Cy, Cz, Cl, Cm, Cn = self.lateral(v, M, alfa, beta, delta, Xcm, p, q, r, launch_vehicle)
            Cx = (Cd*cos(alfa) - 0.5*Cz*sen(2*alfa))/(cos(alfa)**2)
            self.forcecoeff = np.array([Xcp, Cx, Cy, Cz, Cl, Cm, Cn], dtype=object)

    def axial(self, alpha, v, rho, M, launch_vehicle):
      ## Cálculo do Cd para alpha = 0
      # Rocket total length'
      Ltr = launch_vehicle.FusLength + launch_vehicle.NoseLength
      # Mid chord calc (Only for clipped delta plainform geometry)
      thet = launch_vehicle.sweep_angle

      Lm = np.sqrt(launch_vehicle.b**2 + (0.5*launch_vehicle.Ct - 0.5*launch_vehicle.Cr + launch_vehicle.b/tan(thet))**2)
      # Critical reynolds
      Rec = 5E+5

      # Kinematic Viscosity
      ni = 1.789380278077583e-05

      # Reynolds number for each component
      Re_body = rho*v*Ltr/ni
      Re_fin = rho*v*Lm/ni

      B_body = Rec*(0.074/(Re_body**(1/5)) - 1.328/(Re_body**0.5))
      B_fins = Rec*(0.074/(Re_fin**0.25) - 1.328/(Re_fin**0.5))
      # Viscous Friction coefficient calculation
      if Re_body <= Rec:
          Cf_b = 1.328/(Re_body**0.5)
      else:
          Cf_b = 0.074/Re_body**(1/5) - B_body/Re_body

      if Re_fin <= Rec:
          Cf_f = 1.328/(Re_fin**0.5)
      else:
          Cf_f = 0.074/Re_fin**(1/5) - B_fins/Re_fin


      # Body drag
      Cd_body = Cf_b*(1 + 60/(Ltr/launch_vehicle.de)**3 + 0.0025*launch_vehicle.FusLength/launch_vehicle.de)*(2.7*launch_vehicle.NoseLength/launch_vehicle.de + 4*launch_vehicle.FusLength/launch_vehicle.de)

      # Base Drag
      Cd_base = 0.029*(launch_vehicle.db/launch_vehicle.de)**3/Cd_body**0.5

      # Fin Drag
      Afp = launch_vehicle.S + 0.5*launch_vehicle.de*launch_vehicle.Cr
      Cd_fin = 2*Cf_f*(1 + 2*launch_vehicle.tick/Lm)*4*launch_vehicle.N*Afp/(np.pi*launch_vehicle.de**2)

      # Interference Drag
      Cd_i = 2*Cf_f*(1 + 2*launch_vehicle.tick/Lm)*4*launch_vehicle.N*(0.5*launch_vehicle.de*launch_vehicle.Cr)/(np.pi*launch_vehicle.de**2)

      Cd0 = Cd_body + Cd_base + Cd_fin + Cd_i

      ## Cd para alpha > 0
      Cda = 0

      # if alpha*180/np.pi>2:
      #     del_ = self.del_alpha(alpha)
      #     eta_ = self.eta(alpha)
      #     Cdb = 2*del_*alpha**2 + 3.6*eta_*(1.36*Ltr - 0.55*self.NoseLength)*alpha**3/(np.pi*self.de)
      #
      #     # Fin drag
      #     Lts = self.b*2 + self.de
      #     Rs = Lts/self.de
      #
      #     # Interference coefficient
      #     kfb = 0.8065*Rs**2 + 1.1553*Rs
      #     kbf = 0.1935*Rs**2 + 0.8174*Rs + 1
      #
      #     Afp = self.S + 0.5*self.de*self.Cr
      #     Cdf = (alpha**2)*(1.2*Afp*4/(np.pi*self.de**2) + 3.12*(kfb + kbf -1)*(self.S*4/(np.pi*self.de**2)))
      #
      #     Cda = Cdf + Cdb

      # Drag coefficient
      Cd = Cda + Cd0

      if M<0.8:
          B = (1 - M**2)**0.5

      elif M>0.8 and M<1.1:
          B = (1 - 0.8**2)**0.5

      else:
          B = (M**2 - 1)**0.5

      # Zero angle of attack drag coefficient
      Cd = Cd/B
      return Cd

    def lateral(self, v, M, alfa, beta, delta, Xcm, p, q, r, launch_vehicle):
      if alfa == 0:
          alfa = 0.000000001

      ## Cálculo do CP e Cna
      # Nose part ------------------------------------------------------------------------------
      Cna_nose = 2
      Xcp_nose = 0.466*launch_vehicle.NoseLength

      # Fins part ------------------------------------------------------------------------------

      # Mid chord calc
      thet = 53.1*np.pi/180
      Lf = np.sqrt(launch_vehicle.b**2 + (0.5*launch_vehicle.Ct - 0.5*launch_vehicle.Cr + launch_vehicle.b/tan(thet))**2)

      # Cna fin
      Kbf = 1 + launch_vehicle.de*(2*launch_vehicle.b+launch_vehicle.de)**-1 # Interference factor
      Cna_fins = Kbf*(4*launch_vehicle.N*launch_vehicle.b/launch_vehicle.de)*(1 + (1 + (2*Lf/(launch_vehicle.Ct+launch_vehicle.Cr))**2)**0.5)**-1

      # Correção para escoamento compressível
      if M<0.8:
          Cna_fins = Cna_fins*(1 - M**2)**-0.5
      elif M<=1.1 or M>=0.8:
          Cna_fins = Cna_fins*(1-0.8**2)**-0.5
      else:
          Cna_fins = Cna_fins*(M**2-1)**-0.5

      # Fin Cp
      Xf = launch_vehicle.FusLength + launch_vehicle.NoseLength - launch_vehicle.Cr
      Xcp_f = Xf + (Lf/3)*((launch_vehicle.Cr + 2*launch_vehicle.Ct)/(launch_vehicle.Cr+launch_vehicle.Ct)) + (1/6)*(launch_vehicle.Cr + launch_vehicle.Ct - ((launch_vehicle.Cr*launch_vehicle.Ct)/(launch_vehicle.Cr+launch_vehicle.Ct)))

      # Rocket body lift correction ----------------------------------------------------------
      Cna_correction = 1.1 * launch_vehicle.de*(launch_vehicle.FusLength/launch_vehicle.Sbt)*alfa**2
      Cna_L = Cna_correction
      Xcp_body = launch_vehicle.NoseLength + launch_vehicle.FusLength/2

      # Calculation results
      CNa = Cna_fins + Cna_nose + Cna_L
      Xcp = (Cna_fins*Xcp_f + Cna_nose*Xcp_nose + Cna_L*Xcp_body)/(CNa)

      Cy = CNa*beta
      Cz = CNa*alfa
      Xsm = (Xcp - Xcm)/launch_vehicle.de

      ## Coeficientes de MOMENTO
      # Rotação em torno de x
      if delta != 0:
          Clp = -launch_vehicle.AR/(2*np.pi)
      else:
          Clp = 0

      ymac = launch_vehicle.b*(launch_vehicle.Cr+2*launch_vehicle.Ct)/(3*(launch_vehicle.Cr+launch_vehicle.Ct))
      Cl_delta = (ymac+launch_vehicle.de/2)*CNa/launch_vehicle.de
      Cl = Cl_delta*delta-Clp*p*launch_vehicle.de/(2*v**2)

      #         Cl = 2*CN
      # Rotação em torno de y
      Cmq = -CNa*Xsm**2*launch_vehicle.de/v
      Cmap = -np.pi/(9*launch_vehicle.AR)
      Cma = CNa*-Xsm
      Cm = Cma*alfa + (Cmap + Cmq)*q/2

      # Rotação em torno de z
      Cnb = -Cma
      Cnp = -np.pi*alfa/(9*launch_vehicle.AR) - 32*np.pi*q*launch_vehicle.b/(135*v*launch_vehicle.AR**2)
      Cn = Cnb*beta + (Cmq*r +Cnp*p)*launch_vehicle.de/(2*v)
      return Xcp, Cy, Cz, Cl, Cm, Cn

    def eta(self, alpha):
        xx = np.rad2deg(alpha)
        yy = cs_eta(xx)
        if yy>1:
            yy = 1
        return yy

    def del_alpha(self, alpha):
        xx = np.rad2deg(alpha)
        yy = cs_del(xx)
        if yy>1:
            yy = 1
        return yy

class wind():
    def __init__(self):
        self.u_wind = nc.Dataset('wind_model/uwnd.2020.nc')
        self.v_wind = nc.Dataset('wind_model/vwnd.2020.nc')
        # Getting the index of the lat x lon localization (Close to Santa Maria, RS, Brazil)
        lat_index = list(self.u_wind['lat'][:]).index(-30)
        lon_index = list(self.u_wind['lon'][:]).index(307.5)
        # Saving the pressures to calculate the altitude
        pressures = np.array(self.u_wind['level'][:])

        # Calculating the altitude
        self.h = np.array(list((map(calc_high, pressures))))

        # Getting the value of wind speed in x and y direction u and v, respectively
        self.u_wind = self.u_wind['uwnd'][28, :, lat_index, lon_index] # Getting data for late january
        self.v_wind = self.v_wind['vwnd'][28, :, lat_index, lon_index] # Getting data for late january

    def wind_calculation(self):
        #Data and parameter (i.e. prior) standard deviations
        sigma_data = np.std(self.u_wind)/3
        data_noise = np.random.normal(0, sigma_data, 1).reshape(-1,1)[0][0]

        # Creating a training Dataset
        X_train = np.sort(np.random.uniform(0, max(self.h), size=(400, 1)).flatten())
        X_train -= min(X_train)
        X_train[1::] += data_noise*10

        # Modeling for u component
        M = 12
        u_component = wind_component(M, self.h, X_train, self.u_wind)
        v_component = wind_component(M, self.h, X_train, self.v_wind)

        return u_component+data_noise, v_component+data_noise, X_train

class six_dof():
    def __init__(self, sim, launch_vehicle):
        self.rocket = launch_vehicle
        print('Importing propulsion and structural data')
        print('Loading rec information')
        self.is_droguechute     = sim.drogue_chute
        self.is_mainchute       = sim.main_chute
        self.is_motormissalign  = sim.motor_missalign
        self.Cd_drogue          = sim.Cd_drogue
        self.D_drogue           = sim.diameter_drogue
        self.t_drogueopen       = sim.time_open
        self.Cd_main            = sim.Cd_main
        self.D_main             = sim.diameter_main
        self.h_open             = sim.h_open

        print('Loading missalignment information')
        self.eps                = sim.eps
        self.mu                 = sim.mu

        print('Loading wind information')
        self.is_wind = sim.wind
        if self.is_wind:
            u_wind, v_wind, altitude = wind().wind_calculation()
            altitude = np.sort(altitude)
            self.cs_uwind = CubicSpline(altitude, u_wind)
            self.cs_vwind = CubicSpline(altitude, v_wind)

        print('Simulation Resume:')
        self.print_resume(sim.step, sim.plot_trajectory,
                          sim.plot_t_trajectory, sim.save)

        confirmation = input('Proceed with simulation [y/n]')
        if confirmation.lower() == 'n':
            return 0
        else:
            print('Initializing variables')
            self.initialize(sim)

    def initialize(self, sim):
        # Propulsion and structural information
        self.T = self.rocket.T
        self.M = self.rocket.M
        self.Ixx = self.rocket.Ixx
        self.Iyy = self.rocket.Iyy
        self.Izz = self.rocket.Izz
        self.Xcm = self.rocket.Xcm

        # Propulsion burn and step time
        n = len(self.rocket.T)                                                       # Time data Point
        self.burn_time      = max(self.rocket.t)
        self.burn_t         = np.linspace(0, max(self.rocket.t), n)

        # Simulation parameters
        self.tol                = sim.tol
        self.k                  = sim.step
        self.plot_trajectory    = sim.plot_trajectory
        self.plot_trajectory_time  = sim.plot_t_trajectory,
        self.save               = sim.save
        self.save_name          = sim.save_name

        # Initializing the uncertainty
        self.uncertainty()

        # Initializing the model
        self.is_burntime    = False
        self.is_trail       = 1
        self.burn_height    = 0
        self.is_hzero       = False
        self.apogee         = 0
        self.t_apogee       = 0
        self.t_drogue       = 0
        self.t_main         = 0
        self.STM            = np.array([0])                                                # Margem estática
        self.latitude       = np.array([self.rocket.latitude])                          # Latitude inicial
        self.longitude      = np.array([self.rocket.longitude])                        # Longitude inicial
        self.u              = np.array([self.rocket.u0])                                       # Velocidade x inicial (b-ref)
        self.v              = np.array([self.rocket.v0])                                       # Velocidade y inicial (b-ref)
        self.w              = np.array([self.rocket.w0])                                       # Velocidade z inicial (b-ref)
        self.acc            = np.array([self.rocket.u0])                                     # Aceleração (b-ref)
        self.p              = np.array([self.rocket.p0])                                       # Velocidada angular x inicial (b-ref)
        self.q              = np.array([self.rocket.q0])                                       # Velocidada angular y inicial (b-ref)
        self.r              = np.array([self.rocket.r0])                                       # Velocidada angular z inicial (b-ref)
        self.Va             = np.array([self.rocket.Va])                                      # Velocidade aerodinâmica inicial
        self.x              = np.array([self.rocket.x0])                                       # Posição x inicial (h-ref)
        self.y              = np.array([self.rocket.y0])                                       # Posição y inicial (h-ref)
        self.z              = np.array([self.rocket.z0])                                       # Posição z inicial (h-ref)
        self.theta          = np.array([self.rocket.elevation + self.d_elevation])         # Elevação inicial
        self.psi            = np.array([self.rocket.azimute + self.d_azimuth])               # Azimute inicial
        self.phi            = np.array([self.rocket.phi0])                                   # Giro inicial
        self.alpha          = np.array([self.rocket.alpha0])                               # Ângulo de ataque inicial
        self.beta           = np.array([self.rocket.beta0])                                 # Ângulo lateral inicial
        self.delta          = self.rocket.delta                                            # Desalinhamento da aleta
        self.q_dyn          = np.array([0])                                              # Pressão dinâmica
        self.DragCoeff      = np.array([])
        self.Cx             = np.array([])
        self.Cy             = np.array([])
        self.Cz             = np.array([])
        self.Cl             = np.array([])
        self.Cm             = np.array([])
        self.Cn             = np.array([])
        self.t              = np.array([0.0])
        self.Thrust_Interp  = CubicSpline(self.burn_t, self.rocket.T)
        self.Mass_Interp    = CubicSpline(self.burn_t, self.rocket.M)
        self.Xcm_Interp     = CubicSpline(self.burn_t, self.rocket.Xcm)
        self.Ixx_Interp     = CubicSpline(self.burn_t, self.rocket.Ixx)
        self.Iyy_Interp     = CubicSpline(self.burn_t, self.rocket.Iyy)

        print('Initializng solver:')

        self.solve_ascending()

    def uncertainty(self):
        self.d_thrust = 0# np.random.normal()*10 # 10 N de desvio padrão
        self.d_elevation = 0# np.random.normal()*3*np.pi/180 # 3 graus de desvio padrão
        self.d_azimuth = 0# np.random.normal()*3*np.pi/180 # 3 graus de desvio padrão
        self.d_force = 0# np.random.normal()*0 # 5 N de desvio padrão
        self.d_moment = 0# np.random.normal()*0 # 3 N.m de desvio padrão
        self.d_Xcp = 0# np.random.normal()*0 # 3 cm de desvio padrão
        self.d_Ji = 0# np.random.normal()*0 # 4 kg.m^2 de desvio padrão

    def return_acc_asc(self, X, F_ext, M_ext, J, m):
        # u = X[0], v = X[1], w = X[2], p = X[3], q = X[4], r = X[5], psi = X[6], theta = X[7], phi = X[8], x = X[9], y = X[10], z = X[11]
        u = X[0]; v = X[1]; w = X[2]; p = X[3]; q = X[4]; r = X[5]; psi = X[6]; theta = X[7]; phi = X[8]; x = X[9]; y = X[10]; z = X[11];
        Fx = F_ext[0]; Fy = F_ext[1]; Fz = F_ext[2];
        L = M_ext[0]; Ml = M_ext[1]; N = M_ext[2];
        Jx = J[0]; Jy = J[1]; Jz = J[2];

        # Acelerações no referencial do corpo
        up=Fx/m - self.rocket.g0*sen(theta) + r*v - q*w
        vp=Fy/m + self.rocket.g0*cos(theta)*sen(phi) - r*u + p*w
        wp=Fz/m + self.rocket.g0*cos(theta)*cos(phi) + q*u - p*v

        # Solução para x e y
        xp = u*cos(theta)*cos(psi)+v*(-cos(phi)*sen(psi)+sen(phi)*sen(theta)*cos(psi)) + w*(sen(phi)*sen(psi)+cos(phi)*sen(theta)*cos(psi))
        yp = u*cos(theta)*sen(psi)+v*(cos(phi)*cos(psi)+sen(phi)*sen(theta)*sen(psi)) + w*(-sen(phi)*cos(psi)+cos(phi)*sen(theta)*sen(psi))

        # Cinemática de rotação
        psip = (1/cos(theta))*(cos(phi)*r + q*sen(phi))
        thetap = q*cos(phi)-r*sen(phi)
        phip = p+np.tan(theta)*(q*sen(phi)+r*cos(phi))

        # Solução do movimento rotacional
        pp = ((Jy - Jz)*q*r + L)/Jx;
        qp = ((Jz - Jx)*p*r + Ml)/Jy;
        rp = ((Jx - Jy)*q*p + N)/Jz;
        # Nova altitude
        hp = u*sen(theta)-v*sen(phi)*cos(theta)-w*cos(phi)*cos(theta)
        return np.array([up, vp, wp, pp, qp, rp, psip, thetap, phip, xp, yp, hp])

    def return_acc_des(self, X, Thrust, D, m, u_wind, v_wind):
        # X = np.array([self.V[i], self.gam[i], self.phi[i], self.x[i], self.y[i], self.z[i], self.delta[i]])
        V = X[0]; gam = X[1]; phi = X[2]; x = X[3]; y = X[4]; z = X[5]; delta = X[6];

        vp = dvdt(Thrust, D, m, gam, phi, delta, z, self.eps, self.mu)
        gamp = dgam_dt(Thrust, m, V, z, gam, phi, delta, self.eps, self.mu)
        phip = dphidt(Thrust, m, V, z, gam, phi, delta, self.mu)
        xp = dxdt(V, gam, phi, u_wind)
        yp = dydt(V, gam, phi, v_wind)
        zp = dzdt(V, gam)
        deltap = ddelta_dt(V, gam, phi, z)

        return np.array([vp, gamp, phip, xp, yp, zp, deltap])

    def solve_ascending(self):
        i = 0
        # Calculo de algumas incertezas
        # self.uncertainty()

        print('Initializing solver')
        while self.is_hzero==False:

            # Propriedades atmosféricas com base no modelo de U.S. Standard Atmosphere, 1976
            rho = atm.ATMOSPHERE_1976(self.z[i]).rho
            v_sound = atm.ATMOSPHERE_1976(self.z[i]).v_sonic

            if self.is_trail or self.is_wind == False:
                u_wind = 0
                v_wind = 0
            elif self.is_wind:
                u_wind = self.cs_uwind(self.z[i])
                v_wind = self.cs_vwind(self.z[i])

            # Módulo da velocidade no referencial aerodinâmico
            ua = self.u[i]-u_wind
            va = self.v[i]-v_wind
            wa = self.w[i]

            self.Va = np.append(self.Va, np.sqrt(ua**2 + va**2 + wa**2))
            Ma = self.Va[i]/v_sound
            # Determinando o componente de empuxo e a massa do foguete na i-ésima iteração
            # Aqui estou considerando o empuxo no eixo longitudal do foguete (x)
            if self.t[i] > 0.0 and self.t[i] >= self.burn_time:
                # Considerando que o motor queimou todo o propelente
                m = self.M[-1] # Massa final após a queima
                Thrust = 0 # Empuxo zero
                # Momento de inércia
                Jx = self.Ixx[-1]
                Jy = self.Iyy[-1]
                Jz = self.Izz[-1]
                if self.is_burntime == False:
                    self.burn_height = [self.x[i], self.y[i], self.z[i]]
                self.is_burntime = True
                xcm = self.rocket.FusLength + self.rocket.NoseLength - self.Xcm[-1]
                self.t = np.append(self.t, self.t[i] + self.k)
            else:
                # Durante a queima
                Thrust = self.Thrust_Interp(self.t[i]) + self.d_thrust
                m = self.Mass_Interp(self.t[i])
                xcm = self.Xcm_Interp(self.t[i])

                # Momentos de Inércia
                Jx = self.Ixx_Interp(self.t[i])
                Jy = self.Iyy_Interp(self.t[i])
                Jz = Jy
                xcm = self.rocket.FusLength + self.rocket.NoseLength - xcm
                self.t = np.append(self.t, self.t[i] + self.k)

            # Verificação caso o foguete ainda esteja no trilho de lançamento
            # Não havendo a presença de rotação
            if self.z[i] <=  self.rocket.l_trilho:
                # Cálculo dos ângulos aerodinâmicos
                self.alpha = np.append(self.alpha, 0)
                self.beta = np.append(self.beta, 0)

                # Cálculo dos coeficientes aerodinâmicos e centro de pressão
                # Cd - Coef. de arrasto; Cn - Coef. de força normal; Ca - Coef. de força axial
                # Cs - Coef. de força lateral
                Xcp, Cx, Cy, Cz, Cl, Cm, Cn = aerodynamic(self.rocket, self.Va[i], rho, Ma, xcm, self.p[i], self.q[i], self.r[i], self.alpha[i], self.beta[i], self.delta).forcecoeff
                Xcp = Xcp + self.d_Xcp/100
                self.DragCoeff = np.append(self.DragCoeff, Cx)
                # O centro de pressão é calculado em relação a ponta do foguete (ponta do nose)
                # E o cálculo do CM é feito com relação a parte inferior do foguete
                # Portanto, a posição de referência do cm é corrigida para calcular a margem estática
                Cl, Cm, Cn = 0, 0, 0

                # Pressão dinâmica * area
                qS = 0.5*self.rocket.Sbt*rho*self.Va[i]**2

                self.q_dyn = np.append(self.q_dyn, qS/self.rocket.Sbt)

                # Calculando da área efetiva de Arrasto
                Aw = np.pi*(self.rocket.de/2)**2*Cx

                # Forças aerodinâmicas'
                Xa = qS*Cx
                Ya = qS*Cy
                Za = qS*Cz

                # Armazenando os coeficientes
                self.Cx = np.append(self.Cx, Cx)
                self.Cy = np.append(self.Cy, Cy)
                self.Cz = np.append(self.Cz, Cz)
                self.Cl = np.append(self.Cl, Cl)
                self.Cm = np.append(self.Cm, Cm)
                self.Cn = np.append(self.Cn, Cn)

                # Forças externas
                Fx = -Xa + Thrust*cos(self.eps)*cos(self.mu)
                Fy = -Ya + Thrust*sin(self.mu)
                Fz = -Za - Thrust*sin(self.eps)*cos(self.mu)

                # Momentos externos
                L  = 0
                Ml = 0
                N  = 0

                # Processo de integração
                X_state = np.array([self.u[i], self.v[i], self.w[i], 0, 0, 0, self.psi[i], self.theta[i], self.phi[i], self.x[i], self.y[i], self.z[i]])
                k1 = self.k*self.return_acc_asc(X_state, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k2 = self.k*self.return_acc_asc(X_state + 0.25*k1, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k3 = self.k*self.return_acc_asc(X_state + (3/32)*k1 + (9/32)*k2, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k4 = self.k*self.return_acc_asc(X_state + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3 , [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k5 = self.k*self.return_acc_asc(X_state + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k6 = self.k*self.return_acc_asc(X_state - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                delta_45 = (1/360)*k1 - (128/4275)*k2 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6
                delta_45 = np.linalg.norm(delta_45)

                self.k = 0.9*self.k*(self.tol/delta_45)**(1/5)

                X_new  = X_state + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - k5/5
                self.u = np.append(self.u, X_new[0])
                self.v = np.append(self.v, X_new[1])
                self.w = np.append(self.w, X_new[2])
                self.p = np.append(self.p, X_new[3])
                self.q = np.append(self.q, X_new[4])
                self.r = np.append(self.r, X_new[5])
                self.psi = np.append(self.psi, X_new[6])
                self.theta = np.append(self.theta, X_new[7])
                self.phi = np.append(self.phi, X_new[8])
                self.x = np.append(self.x, X_new[9])
                self.y = np.append(self.y, X_new[10])
                self.z = np.append(self.z, X_new[11])

                self.STM = np.append(self.STM, (Xcp - xcm)/self.rocket.de)

                # Taxas de variações
                O_acc = self.return_acc_asc(X_state, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                self.acc = np.append(self.acc, np.sqrt(O_acc[0]**2 + O_acc[1]**2 + O_acc[2]**2))

                ''' ----------------------------------------------------------------  '''
            else:
                if self.is_trail == 1:
                    print('SAIU DO TRILHO------------')
                    self.v_trail = self.Va[i]
                    self.is_trail = 0

                self.alpha = np.append(self.alpha, np.arctan(wa/ua))
                self.beta = np.append(self.beta, np.arcsin(va/self.Va[i]))

                # Cálculo dos coeficientes aerodinâmicos e centro de pressão
                # Cd - Coef. de arrasto; Cn - Coef. de força normal; Ca - Coef. de força axial
                # Cs - Coef. de força lateral
                Xcp, Cx, Cy, Cz, Cl, Cm, Cn = aerodynamic(self.rocket, self.Va[i], rho, Ma, xcm, self.p[i], self.q[i], self.r[i], self.alpha[i], self.beta[i], self.delta).forcecoeff
                Xcp = Xcp + self.d_Xcp/100
                self.DragCoeff = np.append(self.DragCoeff, Cx)

                # Pressão dinâmica * area
                qS = 0.5*self.rocket.Sbt*rho*self.Va[i]**2
                self.q_dyn = np.append(self.q_dyn, qS/self.rocket.Sbt)

                # Calculando da área efetiva de Arrasto
                Aw = np.pi*(self.rocket.de/2)**2*Cx

                # Forças aerodinâmicas'
                Xa = qS*Cx
                Ya = qS*Cy
                Za = qS*Cz

                # Armazenando os coeficientes
                self.Cx = np.append(self.Cx, Cx)
                self.Cy = np.append(self.Cy, Cy)
                self.Cz = np.append(self.Cz, Cz)
                self.Cl = np.append(self.Cl, Cl)
                self.Cm = np.append(self.Cm, Cm)
                self.Cn = np.append(self.Cn, Cn)

                # Forças externas
                Fx = -Xa + Thrust*cos(self.eps)*cos(self.mu)
                Fy = -Ya + Thrust*sin(self.mu)
                Fz = -Za - Thrust*sin(self.eps)*cos(self.mu)

                # Momentos externos
                alav = self.STM[i]*self.rocket.de
                L  = qS*self.rocket.de*Cl + self.d_moment
                Ml = qS*self.rocket.de*Cm + Thrust*sin(self.mu)*alav + self.d_moment
                N  = qS*self.rocket.de*Cn + Thrust*sin(self.eps)*cos(self.mu)*alav + self.d_moment

                # Processo de integração - RK felhberg
                X_state = np.array([self.u[i], self.v[i], self.w[i], self.p[i], self.q[i], self.r[i], self.psi[i], self.theta[i], self.phi[i], self.x[i], self.y[i], self.z[i]])
                k1 = self.k*self.return_acc_asc(X_state, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k2 = self.k*self.return_acc_asc(X_state + 0.25*k1, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k3 = self.k*self.return_acc_asc(X_state + (3/32)*k1 + (9/32)*k2, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k4 = self.k*self.return_acc_asc(X_state + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3 , [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k5 = self.k*self.return_acc_asc(X_state + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                k6 = self.k*self.return_acc_asc(X_state - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)

                delta_45 = (1/360)*k1 - (128/4275)*k2 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6
                delta_45 = np.linalg.norm(delta_45)

                self.k = 0.9*self.k*(self.tol/delta_45)**(1/5)

                X_new = X_state + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - k5/5

                self.u = np.append(self.u, X_new[0])
                self.v = np.append(self.v, X_new[1])
                self.w = np.append(self.w, X_new[2])
                self.p = np.append(self.p, X_new[3])
                self.q = np.append(self.q, X_new[4])
                self.r = np.append(self.r, X_new[5])
                self.psi = np.append(self.psi, X_new[6])
                self.theta = np.append(self.theta, X_new[7])
                self.phi = np.append(self.phi, X_new[8])
                self.x = np.append(self.x, X_new[9])
                self.y = np.append(self.y, X_new[10])
                self.z = np.append(self.z, X_new[11])

                # Margem estática
                self.STM = np.append(self.STM, (Xcp - xcm)/self.rocket.de)

                # Taxas de variações
                O_acc = self.return_acc_asc(X_state, [Fx, Fy, Fz], [L, Ml, N], [Jx, Jy, Jz], m)
                self.acc = np.append(self.acc, np.sqrt(O_acc[0]**2 + O_acc[1]**2 + O_acc[2]**2))

                # Verificando se o apogeu foi atingido (medido no ref fixo na terra)
                if self.z[i+1] < self.z[i] and self.apogee==0:
                    print('Apogee')
                    self.apogee = self.z[i]
                    self.t_apogee = self.t[i]
                    break


                # Tempo de voo limitado para 200 seg (de acorpo com o open rocket é em torno de 150 seg para esse foguete)
                if self.t[i]>250:
                    print('Solução divergiu')
                    break;

            i += 1
            print('T: ', Thrust, 'Uwind: ', u_wind, 'Vwind: ', v_wind)
            print('t: ', self.t[i-1], '\nAltitute: ', self.z[i-1], '\nVelocidade: ', self.Va[i-1])
            print('-----------------------------------------------')

        print('Simulation for ascending phase done')
        print('Initializing descending phase...')
        self.i = i+1
        self.i_ascending = i
        self.solve_descending()

    def solve_descending(self):
        i = self.i
        self.lamb = np.zeros(i)
        self.lamb = np.append(self.lamb, self.rocket.longitude)
        self.delta = np.zeros(i)
        self.delta = np.append(self.delta, self.rocket.latitude)
        self.gam = np.zeros(i)
        self.gam = np.append(self.gam, self.theta[i])                       # Ângulo de voo
        self.phi_ascending = self.phi
        self.phi = np.zeros(i)
        self.phi = np.append(self.phi, self.psi[i])                                             # Ângulo do azimut
        self.V = np.sqrt(self.u**2 + self.v**2 +self.w**2)                                      # Velocidade
        while self.is_hzero==False:
            # Atmospheric properties for r-th altitude
            rho = atm.ATMOSPHERE_1976(self.z[i]).rho
            v_sound = atm.ATMOSPHERE_1976(self.z[i]).v_sonic

            Ma = self.V[i]/v_sound
            # Calculando o Cd com base na altura e velocidade atual
            _, Cd, _, _, _, _, _ = aerodynamic(self.rocket, self.V[i], rho, Ma, 0, 0, 0, 0, 0, 0, 0).forcecoeff

            # Calculando da área efetiva de Arrasto
            if self.is_burntime==True and self.t[i] >= self.t_drogue and self.t_drogue != 0 and self.is_mainchute and self.z[i]<=self.h_open:
                if self.t_main == 0:
                    print('Abrindo Main')
                    self.t_main = self.t[i]
                Aw = np.pi*(self.rocket.de/2)**2*Cd  + np.pi*(self.D_main/2)**2*self.Cd_main
            elif self.t_apogee > 0 and self.t[i] >= self.t_apogee + self.t_drogueopen and self.is_droguechute:
                if self.t_drogue == 0:
                    print('Abrindo drogue')
                    self.t_drogue = self. t[i]
                Aw = np.pi*(self.rocket.de/2)**2*Cd + np.pi*(self.D_drogue**2)*0.25*self.Cd_drogue
            else:
                Aw = np.pi*(self.rocket.de/2)**2*Cd

            # Arrasto
            if self.V[i] < 0:
                f = -1
            else:
                f = 1
            D = drag(rho, self.V[i], Aw)

            Thrust = 0
            m = self.M[-1]
            self.t = np.append(self.t, self.t[i] + self.k)
            self.is_burntime = True

            # Wind calculations
            if self.is_wind == True:
                u_wind = self.cs_uwind(self.z[i])
                v_wind = self.cs_vwind(self.z[i])
            else:
                u_wind = 0
                v_wind = 0

            # Processo de integração
            X_state = np.array([self.V[i], self.gam[i], self.phi[i], self.x[i], self.y[i], self.z[i], self.delta[i]])
            k1 = self.k*self.return_acc_des(X_state, Thrust, D, m, u_wind, v_wind)
            k2 = self.k*self.return_acc_des(X_state + 0.25*k1, Thrust, D, m, u_wind, v_wind)
            k3 = self.k*self.return_acc_des(X_state + (3/32)*k1 + (9/32)*k2, Thrust, D, m, u_wind, v_wind)
            k4 = self.k*self.return_acc_des(X_state + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3, Thrust, D, m, u_wind, v_wind)
            k5 = self.k*self.return_acc_des(X_state + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4, Thrust, D, m, u_wind, v_wind)
            k6 = self.k*self.return_acc_des(X_state - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5, Thrust, D, m, u_wind, v_wind)
            delta_45 = (1/360)*k1 - (128/4275)*k2 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6
            delta_45 = np.linalg.norm(delta_45)

            self.k = 0.9*self.k*(self.tol/delta_45)**(1/5)

            X_new = X_state + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - k5/5
            self.V = np.append(self.V, X_new[0])
            self.gam = np.append(self.gam, X_new[1])
            self.phi = np.append(self.phi, X_new[2])
            self.x = np.append(self.x, X_new[3])
            self.y = np.append(self.y, X_new[4])
            self.z = np.append(self.z, X_new[5])
            self.delta = np.append(self.delta, X_new[6])

            # Taxas de variações
            O_acc = self.return_acc_des(X_state, Thrust, D, m, u_wind, v_wind)
            self.acc = np.append(self.acc, O_acc[0])

            # Caso tenha pousado sair do loop
            if self.z[i+1] <= 0.0 and self.apogee != 0:
                print('Pousou')
                self.is_hzero = True
                break

            # Restriçao de voo para 500 segundos
            if self.t[i]>250:
                print('Soluçao divergiu')
                break;

            i += 1
            print('T: ', Thrust, 'D: ', D, 'Uwind: ', u_wind, 'Vwind: ', v_wind)
            print('Aw:', Aw)
            print('t: ', self.t[i-1], '\nAltitute: ', self.z[i-1], '\nVelocidade: ', self.V[i-1])
            print('-----------------------------------------------')
        print('Simulation Done')

        self.print_results()
        if self.save:
            print('Saving results in a sheet file')
            self.save_results(self.save_name)
        if self.plot_trajectory:
            print('Ploting 3D trajectory')
            self.plot_trajectory_func()
        if self.plot_trajectory_time:
            print('Ploting output as function of time')
            self.plot_trajectory_time_func()

    def print_resume(self, step, plot_trajectory, plot_trajectory_time, save):
        data = '''
        SIMULATION CONFIGURATION------------- \n
        With wind = {} \n
        With drogue = {} \n
        With main = {} \n
        With missalignment = {} \n
        Step size = {} \n
        OUTPUT CONFIGURATION----------------- \n
        Plot trajectory (x, y, z) = {} \n
        Plot time trajectory (t, y) = {} \n
        Save output = {} \n
        DROGUE PARACHUTE DATA---------------- \n
        Cd = {} \n
        Diameter = {}m \n
        Open after apogee = {:.3f}s \n
        MAIN PARACHUTE DATA------------------ \n
        Cd = {} \n
        Diameter = {}m \n
        Open after apogee = {:.3f}m \n
        MOTOR MISSALINGMENT------------------ \n
        Mu = {}º \n
        Eps = {}º \n
        ROCKET CONFIGURATION----------------- \n
        Initial Mass = {} kg \n
        Final Mass = {} kg \n
        Mean Thrust = {} N\n
        '''.format(self.is_wind, self.is_droguechute, self.is_mainchute, self.is_motormissalign,
                   step, plot_trajectory, plot_trajectory_time, save, self.Cd_drogue, self.D_drogue, self.t_drogueopen,
                   self.Cd_main, self.D_main, self.h_open, np.rad2deg(self.mu), np.rad2deg(self.eps), self.rocket.M[0], self.rocket.M[-1], np.mean(self.rocket.T))
        print(data)
        return 1

    def print_results(self):
        data = '''
        Maximum Velocity = {} [m/s]\n
        Max Q = {} [KPa]\n
        Apogee = {} [m]\n
        Time to Apogee = {} [s]\n
        Velocity at landing = {} [m/s]\n
        Flight time = {} [s]\n
        Velocity off rail = {} [m/s]\n
        Burn end coordinate = {} [m]\n
        '''.format(max(self.Va), max(self.q_dyn)/1000, max(self.z), self.t_apogee, self.V[-1], max(self.t), self.v_trail, self.burn_height)
        print(data)

    def plot_trajectory_func(self):
        # Trajetoria 3d
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=self.x[0:self.i_ascending],y=self.y[0:self.i_ascending],z=self.z[0:self.i_ascending],
                        marker=dict(size=0), name = "Ascending Phase - 6DOF Model",
                        line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter3d(x=self.x[self.i_ascending:len(self.x)],y=self.y[self.i_ascending:len(self.x)],z=self.z[self.i_ascending:len(self.x)],
                        marker=dict(size=0), name = "Descending Phase - 3DOF Model",
                        line=dict(color='royalblue', width=2)))

        fig.update_layout(
            title={
                  'text': "Rocket Nominal Trajectory",
                  'y':0.9,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
            scene=dict(xaxis_title="North [m]",
            yaxis_title="East [m]",
            zaxis_title="Altitude [m]"),
            font=dict(
                family="Times New Roman",
                size=18,
                color = "Black",
            )
        )

        fig.update_layout(scene = dict(
                            xaxis = dict(
                                 backgroundcolor="rgb(255, 255, 255)",
                                 gridcolor="black",
                                 gridwidth = 0.5,
                                 showbackground=True,
                                 zerolinecolor="white",),
                            yaxis = dict(
                                backgroundcolor="rgb(255, 255, 255)",
                                gridcolor="black",
                                gridwidth = 0.5,
                                showbackground=True,
                                zerolinecolor="white"),
                            zaxis = dict(
                                backgroundcolor="rgb(255, 255, 255)",
                                gridcolor="black",
                                gridwidth = 0.5,
                                showbackground=True,
                                zerolinecolor="white",),)
                          )
        fig.update_layout(width=2000, height=1000, showlegend=True)
        fig.show()

    def plot_trajectory_time_func(self):
        fig1 = make_subplots(rows=2, cols=2)
        fig1.add_trace(go.Scatter(x=self.t, y=self.z, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=self.t, y=self.V, line=dict(color='royalblue', width=2)), row=1, col=2)
        fig1.add_trace(go.Scatter(x=self.t, y=self.acc, line=dict(color='royalblue', width=2)), row=2, col=1)
        fig1.add_trace(go.Scatter(x=self.x, y=self.y, line=dict(color='royalblue', width=2)), row=2, col=2)
        fig1.update_xaxes(title_text='t [s]', row=1, col=1)
        fig1.update_xaxes(title_text='t [s]', row=1, col=2)
        fig1.update_xaxes(title_text='t [s]', row=2, col=1)
        fig1.update_xaxes(title_text='x [m]', row=2, col=2)

        fig1.update_yaxes(title_text='Altitude [m]', row=1, col=1)
        fig1.update_yaxes(title_text='Velocidade [m/s]', row=1, col=2)
        fig1.update_yaxes(title_text='Aceleração [m/s²]', row=2, col=1)
        fig1.update_yaxes(title_text='y [m]', row=2, col=2)

        fig1.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig1.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)

        fig1.add_annotation(x=self.t_apogee, y=max(self.z), text="Apogeu = "+str(max(self.z)) + " m",
                           showarrow=False, yshift=10, row = 1, col = 1)
        fig1.add_annotation(x=self.t[-1]/1.25, y=max(self.z)/1.5, text="$t_{apogeu} = "+str(self.t_apogee) + " s$",
                           showarrow=False, yshift=10, row = 1, col = 1)

        fig1.add_annotation(x=self.t[-1]/2, y=max(self.V)/1.25, text="$V_{max} = "+str(max(self.V)) + " m/s$",
                           showarrow=False, yshift=10, row = 1, col = 2)
        fig1.add_annotation(x=self.t[-1]/2, y=max(self.V)/1.5, text="$V_{trilho} = "+str(self.v_trail) + " m/s$",
                           showarrow=False, yshift=10, row = 1, col = 2)
        fig1.add_annotation(x=self.t[-1]/2, y=max(self.V)/2.0, text="$V_{pouso} = "+str(self.V[-1]) + " m/s$",
                           showarrow=False, yshift=10, row = 1, col = 2)

        fig1.add_annotation(x=self.t[-1]/2, y=max(self.acc)/1.25, text="$A_{max} = "+str(max(self.acc)) + " m/s²$",
                           showarrow=False, yshift=10, row = 2, col = 1)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        fig2 = make_subplots(rows=2, cols=3)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.p[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.q[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=1, col=2)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.r[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=1, col=3)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.theta[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=2, col=1)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.phi_ascending[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=2, col=2)
        fig2.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.psi[0:self.i_ascending]*180/np.pi, line=dict(color='royalblue', width=2)), row=2, col=3)

        fig2.update_xaxes(title_text='t [s]', row=1, col=1)
        fig2.update_xaxes(title_text='t [s]', row=1, col=2)
        fig2.update_xaxes(title_text='t [s]', row=1, col=3)
        fig2.update_xaxes(title_text='t [s]', row=2, col=1)
        fig2.update_xaxes(title_text='t [s]', row=2, col=2)
        fig2.update_xaxes(title_text='t [s]', row=2, col=3)

        fig2.update_yaxes(title_text='p [º/s]', row=1, col=1)
        fig2.update_yaxes(title_text='q [º/s]', row=1, col=2)
        fig2.update_yaxes(title_text='r [º/s]', row=1, col=3)
        fig2.update_yaxes(title_text=r'$\theta \;\; [º]$', row=2, col=1)
        fig2.update_yaxes(title_text=r'$\phi \;\; [º]$', row=2, col=2)
        fig2.update_yaxes(title_text=r'$\psi \;\; [º]$', row=2, col=3)

        fig2.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig2.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        fig3 = make_subplots(rows=1, cols=2)
        fig3.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.alpha[0:self.i_ascending]*np.pi/180, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.beta[0:self.i_ascending]*np.pi/180, line=dict(color='royalblue', width=2)), row=1, col=2)
        fig3.update_xaxes(title_text='t [s]', row=1, col=1)
        fig3.update_xaxes(title_text='t [s]', row=1, col=2)

        fig3.update_yaxes(title_text=r'$ \alpha \;\; [rad]$', row=1, col=1)
        fig3.update_yaxes(title_text=r'$ \beta \;\; [rad]$', row=1, col=2)

        fig3.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig3.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)


        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        fig4 = make_subplots(rows=1, cols=3)
        fig4.add_trace(go.Scatter(x=self.t, y=self.u, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig4.add_trace(go.Scatter(x=self.t, y=self.v, line=dict(color='royalblue', width=2)), row=1, col=2)
        fig4.add_trace(go.Scatter(x=self.t, y=self.w, line=dict(color='royalblue', width=2)), row=1, col=3)
        fig4.update_xaxes(title_text='t [s]', row=1, col=1)
        fig4.update_xaxes(title_text='t [s]', row=1, col=2)
        fig4.update_xaxes(title_text='t [s]', row=1, col=3)
        fig4.update_yaxes(title_text='u [m/s]', row=1, col=1)
        fig4.update_yaxes(title_text='v [m/s]', row=1, col=2)
        fig4.update_yaxes(title_text='w [m/s]', row=1, col=3)

        fig4.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig4.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)


        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        fig5 = make_subplots(rows=2, cols=3)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cx, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cy, line=dict(color='royalblue', width=2)), row=1, col=2)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cz, line=dict(color='royalblue', width=2)), row=1, col=3)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cl, line=dict(color='royalblue', width=2)), row=2, col=1)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cm, line=dict(color='royalblue', width=2)), row=2, col=2)
        fig5.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.Cn, line=dict(color='royalblue', width=2)), row=2, col=3)

        fig5.update_xaxes(title_text='t [s]', row=1, col=1)
        fig5.update_xaxes(title_text='t [s]', row=1, col=2)
        fig5.update_xaxes(title_text='t [s]', row=1, col=3)
        fig5.update_xaxes(title_text='t [s]', row=2, col=1)
        fig5.update_xaxes(title_text='t [s]', row=2, col=2)
        fig5.update_xaxes(title_text='t [s]', row=2, col=3)

        fig5.update_yaxes(title_text='$C_x$', row=1, col=1)
        fig5.update_yaxes(title_text='$C_y$', row=1, col=2)
        fig5.update_yaxes(title_text='$C_z$', row=1, col=3)
        fig5.update_yaxes(title_text='$C_l$', row=2, col=1)
        fig5.update_yaxes(title_text='$C_m$', row=2, col=2)
        fig5.update_yaxes(title_text='$C_n$', row=2, col=3)

        fig5.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig5.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        fig6 = make_subplots(rows=1, cols=1)
        fig6.add_trace(go.Scatter(x=self.t[0:self.i_ascending], y=self.STM, line=dict(color='royalblue', width=2)), row=1, col=1)
        fig6.update_xaxes(title_text='t [s]', row=1, col=1)
        fig6.update_yaxes(title_text='$STM [cal]$', row=1, col=1)
        fig6.update_xaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)
        fig6.update_yaxes(showline=True, showgrid = False, zeroline=False, linewidth=2, linecolor='black', mirror=True)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        fig1.update_layout(width=1400, height=900, plot_bgcolor='rgb(255,255,255)', showlegend=False,
                            title={'text': "Resultados da simulação",
                                  'y':0.95,
                                  'x':0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'},
                            font=dict(family="Times New Roman",size=16,color = "Black"))
        fig2.update_layout(width=1400, height=900, plot_bgcolor='rgb(255,255,255)', showlegend=False,
                            title={'text': "Movimento Rotacional",
                                  'y':0.95,
                                  'x':0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'},
                            font=dict(family="Times New Roman",size=16,color = "Black"))
        fig3.update_layout(width=1200, height=600, plot_bgcolor='rgb(255,255,255)', showlegend=False,
                            title={'text': "Ângulos Aerodinâmicos (Ângulo de ataque e lateral)",
                                  'y':0.95,
                                  'x':0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'},
                            font=dict(family="Times New Roman",size=16,color = "Black"))
        fig4.update_layout(width=1500, height=600, plot_bgcolor='rgb(255,255,255)', showlegend=False,
                            title={'text': "Movimento Translacional",
                                  'y':0.95,
                                  'x':0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'},
                            font=dict(family="Times New Roman",size=16,color = "Black"))
        fig5.update_layout(width=1500, height=900, plot_bgcolor='rgb(255,255,255)', showlegend=False,
                            title={'text': "Coeficientes Aerodinâmicos",
                                  'y':0.95,
                                  'x':0.5,
                                  'xanchor': 'center',
                                  'yanchor': 'top'},
                            font=dict(family="Times New Roman",size=16,color = "Black"))

        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()

    def save_results(self, save_name):
        output = pd.DataFrame()
        output['t'] = self.t
        output['x'] = self.x
        output['y'] = self.y
        output['z'] = self.z
        print('Saving trajectory solution')
        output.to_excel('trajectory_solution_'+save_name)

        print('Saving 6DOF parameters ')
        output = pd.DataFrame()
        output['t'] = self.t[0:self.i_ascending]
        output['u'] = self.u[0:self.i_ascending]
        output['v'] = self.v[0:self.i_ascending]
        output['w'] = self.w[0:self.i_ascending]
        output['p'] = self.p[0:self.i_ascending]
        output['q'] = self.q[0:self.i_ascending]
        output['r'] = self.r[0:self.i_ascending]
        output['SM'] = self.STM[0:self.i_ascending]
        output['alpha'] = self.alpha[0:self.i_ascending]
        output['beta'] = self.beta[0:self.i_ascending]
        output['theta'] = self.theta[0:self.i_ascending]
        output['phi'] = self.phi_ascending[0:self.i_ascending]
        output['psi'] = self.psi[0:self.i_ascending]
        output.to_excel('6dof_solution'+save_name)

        return 1

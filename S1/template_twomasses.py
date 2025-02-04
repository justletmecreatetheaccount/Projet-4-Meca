#
# LEPL1504 - -    MISSION  -   Modelling a two-mass system
#
# @date 2025
# @author Robotran team
# 
# Universite catholique de Louvain


# import useful modules and functions
from math import sin, cos, pi
import numpy as np
from scipy.integrate import solve_ivp
from scipy.misc import derivative

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class MBSData:
    """ Class to define the parameters of the the double mass-spring model
    and of the problem to solve.
     
    It contains the following fields (in SI units): 
    
    general parameters:
    -------------------
    g:     Gravity
    
    masses:
    -------
    m1:    Unsprung mass
    m2:    Sprung mass
    
    parameters of the tyre:
    -----------------------
    k01:   Stiffness
    d1:    Damping
    z01:   Neutral length
    
    parameters of the suspension:
    -----------------------------
    k02:   Stiffness
    d2:    Damping
    z02:   Neutral length   
    
    parameter of the external force:
    --------------------------------
    Fmax:  Semi-amplitude of the force
    Zmax:  Semi-amplitude of the displacement
    f0:    Frequency at start
    t0:    Time for specifying frequency at start
    f1:    Frequency at end
    t1:    Time for specifying frequency at end
    
    equilibrium positions:
    ----------------------
    q1:    Equilibrium position coordinate of m1
    q2:    Equilibrium position coordinate of m2
    """
    
    def __init__(self):
        self.g = 9.81
        self.m1 = 25
        self.m2 = 315
        self.k01 = 190000
        self.d1 = 107
        self.z01 = 0.375
        self.k02 = 37000
        self.d2 = 4000
        self.z02 = 0.8
        self.t0 = 0
        self.t1 = 10
        self.f0 =1
        self.f1 = 10
        self.Fmax = 10000

        self.q1 = 0.357445263
        self.q2 = 0.716482432

        
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def sweep(t, t0, f0, t1, f1, Fmax):
    """ Compute the value of a force sweep function at the given time.
    The sweep function has a sinusoidal shape with constant amplitude 
    and a varying frequency. This function enables to consider linear
    variation of the frequency between f0 and f1 defined at time t0 and t1.

	:param t: the time instant when to compute the function.
	:param t0: the time instant when to specify f0.
	:param f0: the frequency at time t0.
	:param t1: the time instant when to specify f1.
	:param f1: the frequency at time t1.
	:param Fmax: the semi-amplitude of the function.
		
	:return Fext: the value of the sweep function.
    """
    return Fmax* sin(2*pi*(f0 + (((f1-f0)/(t1-t0))*t/2))*t)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    
def compute_derivatives(t, y, data):
    """ Compute the derivatives yd for a given state y of the system.
        The derivatives are computed at the given time t with
        the parameters values in the given data structure.
        
        It is assumed that the state vector y contains the following states:
          y = [q1, q2, qd1, qd2] with:
             - q1: the mass 1 position
             - q2: the mass 2 position 
             - qd1: the mass 1 velocity
             - qd2: the mass 2 velocity 

        :param  t: the time instant when to compute the derivatives.
        :param  y: the numpy array containing the states 
        :return: yd a numpy array containing the states derivatives  yd = [qd1, qd2, qdd1, qdd2]
        :param data: the MBSData object containing the parameters of the model
    """                 
    Fext = sweep(t, data.t0, data.f0, data.t1, data.f1, data.Fmax)
    q1 = y[0]
    q2 = y[1]
    qd1 = y[2]
    qd2 = y[3]

    q2dd = (- Fext + data.k02*(data.z02 - q2) - data.d2*(qd2 - qd1) - data.m2 * data.g)/data.m2
    q1dd = (data.k01*(data.z01 - q1) - data.d1*(qd1) - data.k02*(data.z02 - q2) + data.d2*(qd2 - qd1) - data.m1 * data.g)/data.m1
    return np.array([qd1, qd2, q1dd, q2dd])
    

    # sweep function should be called here: sweep(t, data.t0, data.f0, data.t1, data.f1, data.Fmax)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def compute_dynamic_response(data):
    """  Compute the time evolution of the dynamic response of the two-mass system
         for the given data. Initial and final time are determined
         by the t0 and t1 parameter of the parameter data structure.
         Results are saved to three text files named dirdyn_q.res, dirdyn_qd.res and dirdyn_qdd.res
 
        Time evolution is computed using an time integrator (typically Runge-Kutta).
 
       :param data: the MBSData object containing the parameters of the model
     """
    fprime = lambda t,y: compute_derivatives(t, y, data)
    # ### Runge Kutta ###   should be called via solve_ivp()
    # to pass the MBSData object to compute_derivative function in solve_ivp, you may use lambda mechanism:
    #
    #    fprime = lambda t,y: compute_derivatives(t, y, data)
    #
    # fprime can be viewed as a function that takes two arguments: t, y
    # this fprime function can be provided to solve_ivp
    # Note that you can change the tolerances with rtol and atol options (see online solve_iv doc)
    #
    # Write some code here
    y = solve_ivp(fprime, (data.t0, data.t1), [data.q1, data.q2, 0, 0], t_eval=np.linspace(data.t0, data.t1, 1000))
    

  


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Main function

if __name__ == '__main__':
    mbs_data = MBSData()
    
    compute_dynamic_response(mbs_data)  

"""
modules for quantum dynamics

functions
SOD: second-order difference
RK4: Runge-Kutta 4 integrator
Leap-frog integrator 
Euler method
"""
def euler(a, fun, dt, *args):

    a = a + fun(a, *args) * dt 

    return a

def rk4_step(rho, fun, dt, *args):
    """
    Runge-Kutta 4 integrator to propagate for a single time step
    """
    dt2 = dt/2.0

    k1 = fun(rho, *args )
    k2 = fun(rho + k1*dt2, *args)
    k3 = fun(rho + k2*dt2, *args)
    k4 = fun(rho + k3*dt, *args)

    rho += (k1 + 2*k2 + 2*k3 + k4)/6. * dt

    return rho


def sod(x0, Nt, dt, func, *args):
   """
   second-order difference integration for first-order ODE
   dx/dt = f(x)
   input:
       x0: initial value
       dt: timestep
       Nt: time steps
       func: right-hand side of ODE, f(x)
   """
   dt2 = dt/2.0

   # first-step
   x_half = x0 + func(x0, *args) * dt2
   x1 = x0 + func(x_half, *args)*dt

   xold = x0
   x = x1
   for k in range(Nt):
       xnew = xold + func(x, *args) * 2. * dt

       # update xold
       xold = x
       x = xnew

       # compute observales
       # ...

   return x

#def leapfrog(x0, Nt, dt, func, h0, S1, S2, Lambda1, Lambda2):
#    """
#    leapfrog integration for first-order ODE
#    dx/dt = f(x)
#    input:
#        x0: initial value
#        dt: timestep
#        Nt: time steps
#        func: right-hand side of ODE, f(x)
#    """
#    dt2 = dt/2.0
#
#    # first-step
#    x_half = x0 + func(x0, h0, S1, S2, Lambda1, Lambda2) * dt2
#    x1 = x0 + func(x_half)*dt
#
#
#
#    xold = x0
#    x = x1
#    for k in range(Nt):
#        xnew = xold + func(x, h0, S1, S2, Lambda1, Lambda2) * 2. * dt
#
#        # update xold
#        xold = x
#        x = xnew
#
#        obs(x)
#
#    return x
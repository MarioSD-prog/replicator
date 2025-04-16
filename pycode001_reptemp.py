import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

'''
Simulacion de Ecuacion Replicadora 2 por 2
'''

# Obtencion de la matriz de pago

def matrix_payoff():

    '''
    Gets a 2x2 payoff matrix from user input
    '''

    ent = ['R','S','T','P']
    payoff = []

    # Payoff matrix entries
    for i in range(4):
        e = float(input(f"Payment {ent[i]} = "))
        payoff.append(e)

    # Payoff matrix
    A = np.array(payoff).reshape(2,2)

    return A

# Ecuaci0n Replicadora

def replicator_dynamics(x, A):

    '''
    Calculates the rate of change in population proportions using
    replicator dynamics
    '''

    fitness = np.dot(A,x)
    avg_fitness = np.dot(fitness,x)
    dxdt = x*(fitness - avg_fitness)

    return np.array(dxdt)

# Simulacion

def run_simulation(A, n, xC_ini, tf = 20):

    '''
    Runs the simulation and returns time and population proportions

    Args: A payoff matrix
          n iterations
          xC_ini initial proportion of the first strategy
          tf final time

    Returns: time and population proportions
    '''

    t = np.linspace(0, tf, n)
    dt = tf/(n-1)
    x = np.array([xC_ini, 1 - xC_ini])
    x_history = np.zeros((n, 2))
    x_history[0,:] = x

    for i in range(n - 1):
        dxdt = replicator_dynamics(x, A)
        x = x + dxdt*dt
        x_history[i+1,:] = x

    return t, x_history

# Resultados

def plot_results(t, x_history):

    '''
    Plots the simulation results
    '''
    plt.figure(figsize=(8, 4))

    strategy = ['C', 'D']

    for i in range(2):
        plt.plot(t, x_history[:,i], label = f'$x_{i+1}$, $s_{i+1}={strategy[i]}$')

    plt.xlim(0,10)
    plt.ylim(-0.05,1.05)

    plt.xlabel(r'tiempo $t$')
    plt.ylabel(r'Proporcion $x(t),\ x_i=n_i/N$')
    plt.title(r'Ecuacion replicadora temporal')

    plt.legend(loc='upper right')
    #plt.savefig('replicadora06.png')
    plt.show()

# Funcion para indicar el nombre del juego

def game_name(M):
  R = M[0,0]
  S = M[0,1]
  T = M[1,0]
  P = M[1,1]

  if T > R > P > S:
    name = 'Prisoners Dilemma (PD)'
  elif R > T > P > S:
    name = 'Stag-Hunt (SH)'
  elif T > R > S > P:
    name = 'Hawk-Dove (HD)'
  else:
    name = 'Another'
  return name

# Clasificacion RSTP

def RSTP_classification(M):
  R = M[0,0]
  S = M[0,1]
  T = M[1,0]
  P = M[1,1]

  dic = {'R':R,'S':S,'T':T,'P':P}
  vec = np.array([R,S,T,P])
  vec_sort = np.sort(vec)[::-1]

  clave_sort = []
  for i in range(len(vec_sort)):
    clave = [k for k, v in dic.items() if v == vec_sort[i]]
    clave_sort.append(clave)

  return clave_sort

# Equilibrio de Nash

def nash_equilibrium(M):

  '''
  Calculates the Nash equilibrium of a game

  Args: M payoff matrix

  Returns: Nash equilibrium
  '''

  A_p1 = M
  A_p2 = M.T

  for p2 in ['C','D']:
    if p2 == 'C':
      if A_p1[0,0] > A_p1[1,0]:
        brp1C = [0,'C',A_p1[0,0]]   # [suposicion p2,best response p1, pago p1]
      else:
        brp1C = [1,'D',A_p1[1,0]]
    if p2 == 'D':
      if A_p1[0,1] > A_p1[1,1]:
        brp1D = [2,'C',A_p1[0,1]]
      else:
        brp1D = [3,'D',A_p1[1,1]]

  for p1 in ['C','D']:
    if p1 == 'C':
      if A_p2[0,0] > A_p2[0,1]:
        brp2C = [0,'C',A_p2[0,0]]
      else:
        brp2C = [2,'D',A_p2[0,1]]
    if p1 == 'D':
      if A_p2[1,0] > A_p2[1,1]:
        brp2D = [1,'C',A_p2[1,0]]
      else:
        brp2D = [3,'D',A_p2[1,1]]

  brp1 = [brp1C, brp1D]
  brp2 = [brp2C, brp2D]

  nashe = []
  for pos1 in range(2):
    for pos2 in range(2):
      if brp1[pos1][0] == brp2[pos2][0]:
        nashe.append([[brp1[pos1][1],brp2[pos2][1]],[brp1[pos1][2],brp2[pos2][2]]])

  return nashe

# Ejecucion

def run_game():

  # Obtencion de la matriz de pago
  A_pay = matrix_payoff()

  # Funcion para actualizar el grafico
  def update_plot(xC_ini):
    print(f'[x_1,x_2] = [{round(xC_ini,2)},{round(1 - xC_ini,2)}]')
    t, x_his = run_simulation(A_pay, 10000, xC_ini, 10)
    plot_results(t, x_his)

  # Nombre del juego
  gn = game_name(A_pay)
  print(f'Game: {gn}.')

  # Clasificacion del juego
  crstp = RSTP_classification(A_pay)
  print(f'Classification: {crstp[0][0]} > {crstp[1][0]} > {crstp[2][0]} > {crstp[3][0]}.')

  # Equilibrio de Nash
  ne = nash_equilibrium(A_pay)
  for i in range(len(ne)):
    print(f"Nash Eq {i+1}: {ne[i][0]}. Pay {i+1}: {[float(x) for x in ne[i][1]]}.")

  # Widgets para los rangos
  xC_ini_slider = widgets.FloatSlider(value=0.56, min=0.01, max=0.99, step=0.01, description='x_1: ')

  # Interaccion
  wii = widgets.interactive(update_plot, xC_ini=xC_ini_slider)

  display(wii)

# Ejecutar el juego
# run_game()
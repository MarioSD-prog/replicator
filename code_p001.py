import numpy as np

def replicator_dynamics(x, A):
    '''
    Calculates the rate of change in population proportions using
    replicator dynamics
    '''

    fitness = np.dot(A,x)
    avg_fitness = np.dot(fitness,x)
    dxdt = x*(fitness - avg_fitness)

    return np.array(dxdt)
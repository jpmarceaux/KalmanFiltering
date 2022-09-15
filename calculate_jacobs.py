import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pygsti
import pickle 
import numpy as np
import random

def vector_from_outcomes(outcomes, num_outcomes):
    vecout = np.zeros((num_outcomes))
    for key in outcomes.keys():
        vecout[int(key[0], 2)] = outcomes[key]
    return(vecout)

def matrix_from_jacob(jacob, num_outcomes):
    matout = np.zeros((num_outcomes, len(jacob['0'*int(np.log2(num_outcomes))])))
    for key in jacob.keys():
        matout[int(key[0], 2), :] = np.array(jacob[key])
    return matout

def pickle_save(obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def pickle_load(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)
    
output_jdict = dict()

circ_list = pickle_load('circuits_to_process')
model = pickle_load('model_for_processing')

random.shuffle(circ_list)
my_circuits = circ_list[rank::size][0:2]

for ind, circ in enumerate(my_circuits):
    #print(f"Rank {rank} circ_index {ind+1} of {len(my_circuits)} running circuit\n{circ}" )
    output_jdict[circ] = matrix_from_jacob(target_model.sim.dprobs(circ), 2**circ.width)

gathered_jdict = comm.gather(my_jdict, root=0)
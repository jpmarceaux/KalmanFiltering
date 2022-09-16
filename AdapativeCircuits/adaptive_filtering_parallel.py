SEED = 2022

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
from tqdm import tqdm
import numpy as np
import sys
sys.path.append( '/home/jpmarceaux/Simulations/KalmanFiltering_Sandia2021' )
from kalman_gst import *
import random

def pickle_dict(obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_dict(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

from pygsti.modelpacks import smq1Q_XYZI as std
edesign = std.create_gst_experiment_design(8)
circ_list = edesign.circuit_lists[-1]


# setup the datagen model
max_fogi_error_rate = 0.1
mdl_datagen = std.target_model('H+s')
basis1q = pygsti.baseobjs.Basis.cast('pp', 4)
gauge_basis = pygsti.baseobjs.CompleteElementaryErrorgenBasis(
                        basis1q, mdl_datagen.state_space, elementary_errorgen_types='HS')
mdl_datagen.setup_fogi(gauge_basis, None, None, reparameterize=True,
                     dependent_fogi_action='drop', include_spam=True)
ar = mdl_datagen.fogi_errorgen_components_array(include_fogv=False, normalized_elem_gens=True)
target_model = mdl_datagen.copy()
np.random.seed(SEED)
ar = max_fogi_error_rate * np.random.rand(len(ar))
mdl_datagen.set_fogi_errorgen_components_array(ar, include_fogv=False, normalized_elem_gens=True)

# check that the datagen model is CPTP and print metrics w.r.t. the target model
print('Model is CPTP... ', model_is_cptp(mdl_datagen))
print('avg. gate-set infidelity: ', avg_gs_infidelity(mdl_datagen, target_model))
print('mean square error: ', mserror(mdl_datagen, target_model))


# setup the filter model
P0 = 0.1*np.eye(target_model.num_params)
zero_ekf = setup_extended(target_model, P0)

# run an adaptive filter
for idx, circ in enumerate(circ_list):
    f = 0
    
jdict = dict()

all_circuits = [circ for circ in edesign.circuit_lists[-1]]
random.seed(SEED)
random.shuffle(all_circuits)
my_circuits = all_circuits[rank::size][0:2]

for ind, circ in enumerate(my_circuits):
    print(f"Rank {rank} circ_index {ind+1} of {len(my_circuits)} running circuit\n{circ}" )
    jdict[circ] = matrix_from_jacob(target_model.sim.dprobs(circ), 2**circ.width)

gathered_jdict = comm.gather(jdict, root=0)
pickle_dict(gathered_jdict, 'IXYZ_jacs')
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

def tensor_from_hessian(hessian, hilbert_dims):
    """
    returns a 3d-array that when dotted into the state returns the jacobian 
    """
    num_params = len(hessian['0'*int(np.log2(hilbert_dims))])
    tensor_out = np.zeros((hilbert_dims, num_params, num_params))
    for key in hessian.keys():
        tensor_out[int(key[0], 2), :, :] = hessian[key]
    return tensor_out

def pickle_dict(obj, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_dict(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

from pygsti.modelpacks import smq1Q_XYZI as std
maxLengths = [1,2,4,8]
target_model = std.target_model('H+s')
basis1q = pygsti.baseobjs.Basis.cast('pp', 4)
gauge_basis = pygsti.baseobjs.CompleteElementaryErrorgenBasis(
                        basis1q, target_model.state_space, elementary_errorgen_types='HS')
target_model.setup_fogi(gauge_basis, None, None, reparameterize=True,
                     dependent_fogi_action='drop', include_spam=True)
edesign = pygsti.protocols.StandardGSTDesign(target_model, std.prep_fiducials(), std.meas_fiducials(), std.germs(), maxLengths)# setup the fogi model

#calculate jacobians and hessians

my_jdict = dict()
my_hdict = dict()

import random

all_circuits = [circ for circ_list in edesign.circuit_lists for circ in circ_list]
random.seed(314)
random.shuffle(all_circuits)

for circ in tqdm(all_circuits):
    my_jdict[circ] = matrix_from_jacob(target_model.sim.dprobs(circ), 2**circ.width)
    my_hdict[circ] = tensor_from_hessian(target_model.sim.hprobs(circ), 2**circ.width)

gathered_jdict = comm.gather(my_jdict, root=0)
gathered_hdict = comm.gather(my_hdict, root=0)

if rank==0:
	jdict = {k: v for d in gathered_jdict for k, v in d.items()}
	hdict = {k: v for d in gathered_hdict for k, v in d.items()}

	pickle_dict(jdict, 'smq1Q_XYZI_jacs')
	pickle_dict(hdict, 'smq1Q_XYZI_hess')


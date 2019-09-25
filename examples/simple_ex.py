# We need access to the geomdecomp module in the
# parent directory, so we add the parent directory 
# to our system PATH
import sys
sys.path.insert(0, '../')

from geomdecomp import AutomorphismGroup
import numpy as np

# Corresponding adj_matrix file: simple_adj.txt

if __name__ == "__main__":
    x = raw_input('Path to matrix file (.txt): ')
    try:
        adj_matrix = np.loadtxt(x)
    except IOError:
        sys.exit('Invalid path specified.')
    
    # Make sure adj_matrix is a python array, not
    # a numpy array. If so, call numpy's tolist() method.
    AutG = AutomorphismGroup(adj_matrix)
    AutG.decomposition(verbose=True, path_to_bliss='../bliss-0.73/')
    
    print 'Graph clusters (orbits): ' + str(AutG.getOrbits())

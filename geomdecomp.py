# To run: cd to sage.sh and type './sage -python AutomorphismGroup.py'.
# If sage is in your path, simply type 'sage AutomorphismGroup.py'

import numpy as np
import re
import sys
import subprocess
from sage.all import *
import time

# Generate numpy matrix for the matrices specified in KRI17
def generateModularFractalMatrix(init_matrix, iterative_steps):
    shape = init_matrix.shape
    
    if shape[0] != shape[1]:
        print('Matrix is not square, exiting.')
        return []
    
    for val in np.nditer(init_matrix):
        if val > 1:
            print('Matrix contains element(s) > 1, exiting.')
            return []
    
    A = init_matrix
    for i in range(iterative_steps - 1):
        A = np.kron(init_matrix, A)
    
    return A

# Generate python array for the matrices specified in SAW17
def generateFractalMatrix(b_init_str, iterative_steps):
    link_pattern = b_init_str
    
    for i in range(iterative_steps - 1):
        link_pattern = link_pattern.replace('0', '0'*len(b_init_str))
        link_pattern = link_pattern.replace('1', b_init_str)
    
    link_pattern = '0' + link_pattern
    link_pattern = [int(digit) for digit in link_pattern]
    
    fractal_matrix = [link_pattern]
    for i in xrange(len(link_pattern) - 1):
        prev_line = fractal_matrix[i]
        fractal_matrix.append(prev_line[-1:] + prev_line[:-1])
    
    return fractal_matrix

class AutomorphismGroup:
    def __init__(self, adj_matrix, dpi=300):
        self.plot_dpi = dpi
        self.aut_size = 0
        self.orbits = []
        self.gen_list = []
        self.adj_matrix = adj_matrix
        self.is_simple = self.isSimpleGraph(adj_matrix)
        
        try:
            self.G = Graph(matrix(adj_matrix), weighted=(not self.is_simple))
        except ValueError:
            sys.exit('Error: You must input a valid (symmetric) adjacency matrix.')
        
        # Start vertices from 1 instead of 0 to match GAP labeling 
        self.G.relabel(lambda i : i + 1)
    
    def setPlotDPI(self, dpi):
        if dpi > 0:
            self.plot_dpi = dpi
        else:
            print('Error: Invalid DPI specified, ignoring.')
    
    # Returns False if any element in array is > 1, else True
    def isSimpleGraph(self, adj_matrix):
        for arr in adj_matrix:
            if any(i > 1 for i in arr):
                return False
        
        return True
        
    # A function that deals with GAP's weird formatting
    def gapToPythonList(self, gapList):
        gapList = gapList[2:-2].replace('\n', '')
        temp = re.findall("\[([^\]]+)\]", gapList)
        
        return [map(int, re.findall(r'\d+', t)) for t in temp]
    
    # Takes in a list of generators for GAP (as strings) and outputs a 
    # multidimensional list of the disjoint generators. This is the partitoning
    # of a generating set S for Aut(G) into subsets S_i (see my presentation for 
    # more information)
    def createDisjointGens(self, gens):
        disjoint = []
        
        if len(gens) == 1:
            return [gens.pop()]
        
        while len(gens) > 0:
            i = 0
            temp = [gens.pop()]
            
            while i < len(gens):
                tempMoved = gap.MovedPoints(temp)
                permMoved = gap.MovedPoints(gens[i])
                
                if bool(set(tempMoved) & set(permMoved)):
                    temp.append(gens.pop(i))
                    i = 0
                else:
                    i += 1
            
            disjoint.append(temp)
        
        return disjoint
    
    def getOrbits(self):
        return self.orbits
        
    def getSize(self):
        return self.aut_size
    
    def getGenList(self):
        return self.gen_list
    
    def getGenCount(self):
        return len(self.gen_list)
    
    # This is the main function of this class. It calls bliss.sh in /bliss-0.73/,
    # extracts the results it gives, and uses GAP to find any isomorphisms to 
    # a symmetric group S_n. It handles both simple and multigraphs, but not 
    # directed graphs. It can be modified to work with directed graphs though, if 
    # you're willing to get your hands dirty. Set verbose=False when you call it to
    # avoid any unneccessary text output. 
    def decomposition(self, verbose=True, path_to_bliss='./bliss-0.73/'):
        disjointGens = [0]
        
        # coloredNodeNum is the index given for each a colored node when we translate 
        # a multigraph into a simple colored graph. Basically it indexes colored nodes, 
        # and colored nodes are always of larger index than non-colored nodes.
        coloredNodeNum = len(self.adj_matrix) + 1
        
        # graphinfo.txt is basically a file that contains the exact same information as 
        # the adjacency matrix text file, but written in bliss's specific format.
        with open(path_to_bliss + '/graphinfo.txt', 'w+') as graphinfo:
            
            if not self.is_simple:
                # Appears inefficient, but is necessary to have the correct file order for bliss
                for index, x in np.ndenumerate(np.triu(self.adj_matrix,0)):
                    if x > 1.0:
                        graphinfo.write('n ' + str(coloredNodeNum) + ' ' + str(int(x)) + '\n')
                        coloredNodeNum = coloredNodeNum + 1
            
            for index, x in np.ndenumerate(np.triu(self.adj_matrix,0)):
                if x == 1.0:
                    graphinfo.write('e ' + str(index[0]+1) + ' ' + str(index[1]+1) + '\n')
            
            coloredNodeNum = len(self.adj_matrix) + 1
            if not self.is_simple:
                for index, x in np.ndenumerate(np.triu(self.adj_matrix,0)):
                    if x > 1.0:
                        graphinfo.write('e ' + str(index[0]+1) + ' ' + str(coloredNodeNum) + '\n')
                        graphinfo.write('e ' + str(index[1]+1) + ' ' + str(coloredNodeNum) + '\n')
                        coloredNodeNum = coloredNodeNum + 1
        
        # Prepend starting line given the information we know from the previous loops
        with open(path_to_bliss + '/graphinfo.txt', 'r+') as graphinfo:
            
            prependedLine = ''
            edgeCount = sum([2 if elem > 1 else int(elem) for arr in np.triu(self.adj_matrix,0) for elem in arr])
            
            if self.is_simple:
                prependedLine = 'p edge ' + str(len(self.adj_matrix)) + ' ' + str(edgeCount) + '\n'
            else:
                prependedLine = 'p edge ' + str(len(self.adj_matrix) + (coloredNodeNum - (len(self.adj_matrix) + 1))) \
                                + ' ' + str(edgeCount) + '\n'
            
            # Save info already written
            graphinfo_data = graphinfo.read()
            graphinfo.seek(0,0)
            
            # Rewrite saved data with prepended line
            graphinfo.write(prependedLine.rstrip('\r\n') + '\n' + graphinfo_data)
        
        bashCommand = './bliss graphinfo.txt'
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd=path_to_bliss)
        output, error = process.communicate()
        
        # Extract information from bliss terminal output
        for line in output.splitlines():
            if 'Generator: ' in line:
                self.gen_list.append(line.split('Generator: ', 1)[1])
            elif '|Aut|: ' in line:
                self.aut_size = int(line.split('|Aut|: ', 1)[1].strip())
        
        if self.aut_size == 1:
            if verbose:
                print('Trivial automorphism group, ending function call ...')
            return
        
        disjointGens = self.createDisjointGens([gen for gen in self.gen_list])
        
        # Aquire the orbits and isomorphisms of the geometric factors via GAP
        geomFactorList = []
        geomFactorOrbits = []
        for index, gen in enumerate(disjointGens):
            geomFactor = gap.Group(gen)
            geomFactorList.append(geomFactor)
            
            orbits = gap.Orbits(geomFactor)
            geomFactorOrbits.append(gap.eval(orbits))
            
            # Determine if motif is a symmetric group
            if verbose:
                minOrbitSize = gap.Minimum(gap.List(orbits, gap.Size))
                isNaturalAction = True
                # check if all orbit sizes are equal for our geometric factor
                if minOrbitSize == gap.Maximum(gap.List(orbits, gap.Size)):
                    for orb in orbits:
                        if not gap.IsNaturalSymmetricGroup(gap.Action(geomFactor, orb)):
                            isNaturalAction = False
                            print('H' + str(index) + ' is a complex motif\n')
                            break
                    
                    if isNaturalAction:
                        print('H' + str(index) + u'\u2245' + ' S' + str(minOrbitSize) + '\n')
                else:
                    print('H' + str(index) + ' is a complex motif\n')
        
        # concatenate lists of orbits
        orbitList = sum((self.gapToPythonList(orb) for orb in geomFactorOrbits), [])
        orbitList = [sorted(orb) for orb in orbitList]
        
        # convert simple colored graph orbits to multigraph orbits if necessary
        self.orbits = filter(None, [list(filter(lambda x : x <= len(self.adj_matrix), orb)) for orb in orbitList])
        
        # If verbose is True (it is by default), then ask whether the user would like to plot / save the graph.
        if verbose:
            save = raw_input('Would you like to save a plot of G? (y/n): ')
            if save == 'y' or save == 'yes':
                plotPath = raw_input('Enter a valid path (including extension): ')
                
                while True:
                    try:
                        P = self.G.plot(vertex_color='white', partition=self.orbits, vertex_labels=True, \
                                        vertex_size=60, edge_labels=(not self.is_simple))
                                        
                        P.save(plotPath, dpi=self.plot_dpi)
                        os.system('display ' + plotPath)
                    except ValueError:
                        plotPath = raw_input('Please enter a file name with valid extension (.pdf, .png, ...): ')
                    else:
                        break


# This is just a simple way of running the program via this file specifically instead of 
# with an external file importing AutomorphismGroup.
if __name__ == "__main__":
    x = raw_input('Path to matrix file (.txt): ')

    try:
        adj_matrix = np.loadtxt(x)
    except IOError:
        sys.exit('Invalid path specified.')
    
    AutG = AutomorphismGroup(adj_matrix)
    AutG.decomposition(verbose=True)
    
    print ''
    print 'Orbits: ' + str(AutG.getOrbits())

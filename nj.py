import numpy as np
from sys import stdin

class NJ:
    """Implementation of the Neighbour-Joining algorithm
    
    """

    def __init__(self, N, taxa, distances):
        """Neighbour-Joining constructor, creates a new
        NJ object
        
        Arguments:
            N {int} -- distance matrix dimension
            taxa {list<str>} -- taxa names
            distances {np.ndarray} -- distance matrix

        Attributes:
            N {int} -- distance matrix dimension
            taxa {list<str>} -- taxa names
            t_set {set<str>} -- set representation of taxa
            d {np.ndarray} -- distance matrix
            T {dict<list<tuple<str, float>>>} -- tree build by neighbour joining algorithm
        """

        self.N = N
        self.taxa = taxa
        self.t_set = set(taxa)
        self.d = distances
        self.T = {}

    def reduce_matrix(self, i, j):
        """Reduces the distance matrix by 1-dimension, 
        deleting rows and columns of given indices and 
        adding one row and one column containing only 0's
        
        Arguments:
            i {int} -- first index to be removed
            j {int} -- second index to be removed
        
        Returns:
            idx [list<int>] -- original indices of the remaining matrix  
        """

        # get the remaining indices we have left after removing i and j
        idx = [k for k in range(self.N) if k not in {i, j}]
        # create new matrix with 1-dimension less than distance matrix
        zeros = np.zeros((self.N-1, self.N-1))
        # fill the new matrix with the remaining distance matrix starting topleft
        # e.g. removing i=0, j=2, idx = 1,3
        #     0 1 2 3
        # 0 | 0 A B C |           | 0 0 0 |          | 0 E 0 |
        # 1 | A 0 D E |   --->    | 0 0 0 |   --->   | E 0 0 |
        # 2 | B D 0 F |           | 0 0 0 |          | 0 0 0 |
        # 3 | C E F 0 |
        zeros[:-1,:-1] = self.d[np.ix_(idx, idx)]
        # set distance matrix to new matrix
        self.d = zeros
        # update N
        self.N = self.N-1
        return idx

    def corrected_distances(self):
        """Calculates corrected distances needed for the Neighbour-Joining
        algorithm
        
        Returns:
            idx [tuple<int,int>] -- indices of one entry with minimum corrected distance
        """

        # calculate all ri exactly once, no need for recomputation
        r = np.zeros(self.N)
        for i in range(self.N):
            r[i] = (1 / (self.N-2)) * np.sum(self.d[i])
        # initialize
        minval = np.Inf
        idx = (-1, -1)
        m = 0
        # since distance matrix is symmetric and has zeros on the diagonal
        # iterate only over lower triangle matrix
        for i in range(self.N):
            # m increases each iteration by one
            for j in range(m):
                # calculate corrected distance 
                val = self.d[i][j] - r[i] - r[j]
                # no need to actually save corrected distance matrix
                # since we are only interested in one with minimum indices
                # there might be more than one such value - arbitrary take the first we find
                if minval > val:
                    minval = val
                    idx = (i, j)
            m = m + 1
        return idx

    def to_newick(self, node, distance = 0.0, depth = 0):
        """Recursively generate newick representation of the NJ tree
        
        Arguments:
            node {str} -- id of the node in the NJ tree, at first call this is the root
        
        Keyword Arguments:
            distance {float} -- distance from node to parent node (default: {0.0})
            depth {int} -- depth of the node in the tree (default: {0})
        
        Returns:
            [str] -- newick representation of the subtree beginning at node
        """

        newick = []
        # no children -> leaf node
        if node not in self.T:
            # return newick representation of leaf node
            return node + ':' + str(distance)
        # not a leaf node - start parenthesis
        newick.append('(')
        # get newick representation of all children recursivly
        for k in self.T[node]:
            newick.append(self.to_newick(k[0], k[1], depth+1) + ',')
        # all children processed, remove last comma and add parenthesis
        newick[-1] = newick[-1][:-1] + ')'
        # root node?
        if depth > 0:
            # no
            return ''.join(newick) + ':' + str(distance)
        else:
            # yes
            return ''.join(newick) + ';'

    def neighbour_joining(self):
        """Actual implementation of the Neighbour-Joining algorithm
        
        Returns:
            [str] -- newick representation of the generated NJ tree
        """

        # increasing numbers used as ids of newly created internal nodes, starting at 1
        number = iter(list(range(1, 1000)))
        # list of remaining ids left to be merged 
        nodes = self.taxa
        # build tree by repeatedly joining sequences with minimal corrected distances
        while self.N > 2:
            # temporary variable used for generating list of remaining nodes for the next iteration
            temp = []
            # get indices of one entry with minimum corrected distance
            i, j = self.corrected_distances()
            # calculate r's for this entry
            ri = (1 / (self.N-2)) * np.sum(self.d[i])
            rj = (1 / (self.N-2)) * np.sum(self.d[j])
            # make a copy of the old distance matrix d
            arr = np.copy(self.d)
            # reduce the distance matrix by i and j, this changes d
            idx = self.reduce_matrix(i, j)
            # k is the index of the new combined node in the new distance matrix 
            k = self.N-1
            # iterate over the remaining nodes after deleting i and j
            # m: index of the remaining node in the new distance matrix
            # m_idx: index of the remaining node in the old distance matrix
            for m, m_idx in enumerate(idx):
                # calculate distance from remaining node to new node
                dist = 0.5 * (arr[i][m_idx] + arr[j][m_idx] - arr[i][j])
                # set the corresponding entries in the new distance matrix to dist
                self.d[k][m] = dist
                self.d[m][k] = dist
                # add this node to list of remaining nodes for the next iteration
                temp.append(nodes[m_idx])
            # calculate the distances from i and j to the new combined node k
            edge_i_k = 0.5 * (arr[i][j] + ri - rj)
            edge_j_k = 0.5 * (arr[i][j] + rj - ri)
            # get an id for the new node
            node = str(next(number))
            # add the new node to the list of remaining nodes
            temp.append(node)
            # add k -> i,j edges to tree
            # tree logic: only parent -> children edges are saved
            self.T[node] = []
            self.T[node].append((nodes[i], edge_i_k))
            self.T[node].append((nodes[j], edge_j_k))
            # update list of remaining nodes
            nodes = temp
        # only two nodes left
        # normally we would add an edge connecting those two remaining nodes (unrooted tree)
        # but since we want a rooted tree, we are adding a root node between them
        # id of the root node
        node = str(next(number))
        # add root node to tree
        self.T[node] = []
        # edges from root node to remaining nodes
        # each edge has length of the distance between them divided by two
        self.T[node].append((nodes[0], self.d[0][1]/2))
        self.T[node].append((nodes[1], self.d[0][1]/2))
        # get the newick representation of this NJ tree and return it
        return self.to_newick(node)

def main():
    """ Reads in distance matrix from user input

    """

    # read in the distance matrix dimensions
    N = int(stdin.readline())
    # new distance matrix
    distances = np.zeros((N,N))
    # new list of taxa
    taxa = []
    # iterate over each line of user input
    for i in range(N):
        # split input into single words
        line = stdin.readline().split()
        # first word is always taxa id
        taxa.append(line[0])
        # iterate over the remaining words
        for j in range(i):
            # read in distance as float
            dist = float(line[j+1])
            # set corresponding entries
            distances[i][j] = dist 
            distances[j][i] = dist
    # create new neighbour-joining object, using dimensions, taxa ids and distances
    nj = NJ(N, taxa, distances)
    # calculate neighbour-joining algorithm and receive newick representation
    newick = nj.neighbour_joining()
    # print newick representation
    print('\n' + newick)

if __name__ == "__main__":
    main()

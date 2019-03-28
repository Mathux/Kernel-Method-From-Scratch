import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, SparseKernel, DataKernel
from src.kernels.kernel import KernelCreate
from src.tools.utils import nb_diff
from src.tools.utils import Parameters
from src.data.trie_dna import MismatchTrie
from scipy.special import binom
from tqdm import tqdm


class MismatchStringKernel(StringKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": False, "sparse": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers):
                phi[j] += 1 * (nb_diff(x_kmer, b) <= self.param.m)
        return phi


class MismatchTrieKernel(TrieKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": True, "sparse": False}
    Trie = MismatchTrie


class MismatchSparseKernel(SparseKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": False, "sparse": True}

    def _compute_phi(self, x):
        phi = {}
        for _, b in enumerate(self.mers):
            for i in range(len(x) - self.param.k + 1):
                xkmer = x[i:i + self.param.k]
                phi[b] = phi.get(b,
                                 0) + 1 * (nb_diff(xkmer, b) <= self.param.m)
        return phi

#class MismatchDirectComputation(DataKernel, metaclass=KernelCreate) :
#    name = "mismatch"
#    defaultParameters = {"k": 2, "m" : 1, "trie": False, "sparse": True, 'direct' : True}
#    
#    def weights(self, k, m, alphabet_size = 4) :
#        if m == 0 :
#            return [1]
#        if m == 1 :
#            return [1+k*(alphabet_size-1), alphabet_size, 2]
#        if m == 2 :
#            return []
#            
#        
#    def extract_kmers(self,x, k) :
#        kmers = {}
#        kmers_full = []
#        for offset in range(len(x) - k + 1) :
#            xkmer = x[offset : offset + k]
#            kmers[xkmer] = kmers.get(xkmer,0) + 1
#            kmers_full.append(xkmer)
#        return kmers, kmers_full
#    
#    def compute_C(self, sx, sy, k, i) :
#        indices = self.sub_indices(k-i, k-1, 0)
#        #print('indices = ', indices)
#        dict_x = self.create_k_i_mers(sx, indices)
#        #print('s_x = ', sx)
#        #print('dict_x = ', dict_x)
#        dict_y = self.create_k_i_mers(sx, indices)
#        C_i = self.M_0(dict_x, dict_y)
#        return C_i 
#    
#    def create_k_i_mers(self,sx, indices) :
#        temp_dict = {}
#        for key, value in sx.items() :
#            for ind in indices :
#                mer = ''.join(np.array(list(key))[ind])
#                temp_dict[mer] = temp_dict.get(mer, 0) + value
#                #temp_dict[mer] = 1
#        return temp_dict
#
#    def sub_indices(self,l, k, i, current = []) :
#            if l == 1 :
#                return [[t] for t in np.linspace(i,k, k-i + 1, dtype = 'int')]
#            else :
#                temp = []
#                for j in range(i, k) :
#                    sub_indices_l = self.sub_indices(l-1, k, j+1)
#                    for sub in sub_indices_l :
#                        temp.append([j] + sub)
#                return temp+current
#                
#        
#
#          
#    def M_0(self, sx, sy) :
#        prod = 0
#        for key,value in sx.items() :
#            if key in sy.keys() : 
#                prod += sy[key]*sx[key]
#        return prod
#    
#    def cumM(self, MM, full = False) :
#        temp = 0
#        for l in range(len(MM)) :
#            if not full :
#                temp += binom(self.param.k - l, len(MM) + 1 - l)*MM[l]
#            else : 
#                temp += MM[l]
#        
#        return temp     
#
#    def compute_M(self, x, y, t) :
#        sx,_ = self.extract_kmers(x, self.param.k)
#        sy,_ = self.extract_kmers(y, self.param.k)
#        M = {0 : self.M_0(sx, sy)}
#        if self.param.k <= t :
#            full_M = True
#        else : 
#            full_M = False
#        for l in range(1, min(t, self.param.k - 1) + 1) :
#            C = self.compute_C(sx ,sy ,self.param.k, l)
#            #print('C = ', C)
#            M[l] =  C - self.cumM(M)
#        if full_M :
#            M[self.param.k] = (len(x) - self.param.k + 1)**2 - self.cumM(M, full = True)
#        return M
#             
#        
#    def distance(self,alpha_1, alpha_2) :
#        d = 0
#        for p in range(len(alpha_1)) :
#            if alpha_1[p] != alpha_2[p] :
#                d +=1
#        return d
#                    
##    
#    def kernel(self, x, y):
#        t = min(2*self.param.m, self.param.k)
#        I = self.weights(self.param.k, self.param.m)
#        M = self.compute_M(x,y,t)
#        #print('m = ', M, 'I = ', I)
#        return np.sum(np.array([I[tt]*M[tt] for tt in range(len(M))]))
    
#    def kernel(self, x, y):
#        t = min(2*self.param.m, self.param.k)
#        I = self.weights(self.param.k, self.param.m)
#        val = 0
#        sx, sx_full = self.extract_kmers(x, self.param.k) 
#        sy, sy_full = self.extract_kmers(y, self.param.k) 
#        self.d_hamm = {}
#        for key_x in sx.keys() :
#            for key_y in sy.keys() :
#                if not ((key_x,key_y) in self.d_hamm.keys() or (key_y,key_x) in self.d_hamm.keys()) :
#                    dist = int(self.distance(key_x, key_y))
#                    if dist <=t :
#                        self.d_hamm[(key_x,key_y)] = int(self.distance(key_x, key_y))
#        for key_x in sx_full:
#            for key_y in sy_full :
#                if (key_x,key_y) in self.d_hamm.keys():
#                    d = self.d_hamm[(key_x,key_y)]
#                    val += I[d]
#                elif (key_y,key_x) in self.d_hamm.keys():
#                    d = self.d_hamm[(key_y,key_x)]
#                    val += I[d]
#
#s            return val

class __MismatchKernel:
    def __init__(self):
        self.defaultParameters = {
            "k": 5,
            'm': 1,
            "trie": True,
            "sparse": False
        }
        self.name = "mismatch"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.sparse:
            return MismatchSparseKernel(dataset, parameters, verbose)
        else:
            if param.trie:
                return MismatchTrieKernel(dataset, parameters, verbose)
            else :
                return MismatchStringKernel(dataset, param, verbose)


MismatchKernel = __MismatchKernel()

if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {"k": 5, "m": 2}

    # from src.tools.test import EasyTest
    # EasyTest(kernels="mismatch", data="seq", dparams=dparams, kparams=kparams)

    from src.tools.test import KernelTest

    parameters = []
    K = 10
    m = 1
    #parameters.append({"k": K, "m" : m, "sparse": True, "trie": False})
    #parameters.append({"k": K, "m" : m, "sparse": False, "trie": False, "direct" : False})
    parameters.append({"k": K,"m" : m, "sparse": False, "trie": True, "small" : True, "nsmall" : 300})
    KernelTest("mismatch", parameters, synth = False)

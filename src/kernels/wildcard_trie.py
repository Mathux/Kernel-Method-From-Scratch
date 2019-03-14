#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:40:11 2019

@author: evrardgarcelon
"""

import numpy as np
from src.kernels.kernel import TrieKernel, KernelCreate


class WildcardTrieKernel(TrieKernel, metaclass=KernelCreate):
    defaultParameters = {"k": 2, 'm' : 1, 'la' : 1}
    
    def k_value(self, x):
        
        leafs = self.get_leaf_nodes(self.trie)
        self.leaf_kgrams_ = dict((leaf.full_label,
                                      dict((index, (len(kgs),leaf.full_label.count('*'))) for 
                                      index, kgs
                                           in leaf.kgrams.items()))
                                     for leaf in leafs)
        k_x = np.zeros(len(self.data))
        for kmer, count1 in self.unique_kmers(x, self.param.k):
            if kmer in self.leaf_kgrams_.keys():
                for j in range(len(self.data.data)):
                    if j in self.leaf_kgrams_[kmer].keys():

                        kgrams, nb_wildcard = self.leaf_kgrams_[kmer][j]
                        k_x[j] += self.param.la**nb_wildcard*(count1 * kgrams)

        return k_x
    
    


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=True, nsmall=200)
        
    kernel = WildcardTrieKernel(data, parameters = {'k' : 3, 'm' : 0})
    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:46:40 2019

@author: evrardgarcelon
"""

class TrieNode(object) :
    
    def __init__(self, label = None, depth = -1, parent = None, count = 0) :
        
        self.depth = depth 
        self.children = {}
        self.parent = parent
        self.count = count
        if self.parent is None :
            self.label = None
        else :
            if self.parent.label is None :
                self.label = label
            else :
                self.label = self.parent.label + label
        
    def add_child(self,child,label) :
        
        self.children[label] = child
    
    def is_leaf(self) :

        return (len(self.children) == 0)
    
    def get_children(self) :
        
        return list(self.children.values())

class Trie(object) :
    
    def __init__(self,string, vocab, k) :
        self.string = string
        self.vocab = vocab
        self.root = TrieNode(count = 1, depth = 0)
        self.l = len(vocab)
        self.node_to_process = [self.root]
        self.k = k
        while len(self.node_to_process) > 0 :
            node = self.node_to_process.pop()
            if node.depth < k :
                self.process_node(node)
    
    
    def process_node(self,node) :
        
        if node.count > 0 :
            for j in range(self.l):
                if not node.label is None : 
                    label = node.label + self.vocab[j]
                else :
                    label = self.vocab[j]
                count = self.string.count(label)
                if  count > 0 and node.depth < self.k:
                    child = TrieNode(label = self.vocab[j], depth = node.depth + 1, parent = node, count = count)
                    node.add_child(child,self.vocab[j])
            if node.depth < self.k :
                self.node_to_process = node.get_children() + self.node_to_process
        
        
    def is_leaf(self,string) :
        is_leaf = True
        node = self.root
        count = 0
        for c in string :
            try :
                node = node.children[c]
                count = node.count
            except KeyError :
                is_leaf = False
                break
        if is_leaf :
            is_leaf = node.is_leaf()
        return is_leaf,count
    
    def m_mistach(self,string) :
        pass

            
 
string = 'ACCCTGCCTACACCGCGGCGGGGACAGGTGGAGGTTTCAACCCCTGTTTGGCAACCTCGGGCGCAGCCAGGCCCCGCCCAGAAATTTCCGGGACACGCCCC'   
vocab  = { 0 : 'A',
           1 : 'T',
           2 : 'G',
           3 : 'C'}
        
if __name__ == '__main__' :
    trie = Trie(string,vocab,3)
    
                
            
            
        

    
    
        
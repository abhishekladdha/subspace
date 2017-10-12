# -*- coding: utf-8 -*-
import os
from collections import Counter
import cPickle as pkl
import json
import logging
import sys
import numpy as np
import errno
import random
import string
import codecs
import math

class prepareData():

	def __init__(self,params):
		self.logger = logging.getLogger('prepare_data')
		self.params = params
		self.unk_symbol = '<unk>'
		self.unk_word_id = 0
		self.data_dir = params['data_dir']
		self.max_entity = params['max_len']
		self.create_vocab()
		
	def read_triples(self):	
		with codecs.open('../wikidata/wikidata_frequent_triple.json','r','utf-8') as data_file:
        		triples = json.load(data_file)
        		print 'Successfully Loaded relevant triple'
        	return triples

	def safe_pickle(self,obj, filename):
		if os.path.isfile(filename):
			self.logger.info("Overwriting %s." % filename)
		else:
			self.logger.info("Saving to %s." % filename)

		try:
        		os.makedirs(os.path.dirname(filename))
    		except OSError as exception:
        		if exception.errno != errno.EEXIST:
            			raise	
		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

	def build_vocab(self,counter,cutoff,vocab_file):
		total_freq = sum(counter.values())
		self.logger.info('Total frequency in dictionary %d', total_freq)
		self.logger.info('Cutoff %d', cutoff)				

		vocab_count = [x for x in counter.most_common() if x[1] > cutoff]
		self.safe_pickle(vocab_count, vocab_file.replace('.pkl','_counter.pkl'))
		#vocab_dict = {self.pad_symbol:self.pad_word_id, self.unk_symbol:self.unk_word_id}
		vocab_dict = {self.unk_symbol:self.unk_word_id}
		for (word, count) in vocab_count:
	                if not word in vocab_dict:
	                	vocab_dict[word] = len(vocab_dict)
	        self.logger.info('Vocab size %d', len(vocab_dict))
		return vocab_dict

	def pad(self, ids, length):
		if len(ids) > length:
			return ids[:length]
		else:
			a = [self.unk_word_id]*(length-len(ids))
			return ids + a

	def convert_ent_type_id(self,ty,ents):
		return [self.lambda_vocab[ty+i] if i in self.entity_vocab else self.unk_word_id for i in ents]

	def convert_ent_id(self,ents):
		return [self.entity_vocab[i] if i in self.entity_vocab else self.unk_word_id for i in ents]	

	def raw_training_data(self, batch_size=200):
        	n_triple = len(self.train_triple)
        	rand_idx = np.random.permutation(n_triple)
        	start = 0
        	while start < n_triple:
            		end = min(start + batch_size, n_triple)
            		#yield self.__train_triple[rand_idx[start:end]]
            		yield self.corrupt_sample(self.train_triple[rand_idx[start:end]])
            		start = end
	
	def testing_data(self, batch_size=200):
        	n_triple = len(self.test_triple)
        	start = 0
        	while start < n_triple:
            		end = min(start + batch_size, n_triple)
            		yield self.test_triple[start:end, :]
            		start = end

    	def validation_data(self, batch_size=200):
        	n_triple = len(self.valid_triple)
        	start = 0
        	while start < n_triple:
            		end = min(start + batch_size, n_triple)
            		yield self.valid_triple[start:end, :]
            		start = end

	def corrupt_sample(self,htr):
        	neg_h_list = []
        	neg_t_list = []
        	for idx in range(htr.shape[0]):
            		prob = float(np.random.sample())
            		#print htr[idx, 2], 
            		#bernoulli_prob = float(self.stat_r[htr[idx, 2]][0])/(self.stat_r[htr[idx, 2]][0] + self.stat_r[htr[idx, 2]][1])
            		bernoulli_prob = 0.5
            		#print prob, bernoulli_prob
            		if(prob < bernoulli_prob):
                		neg_t = -1
                		while neg_t < 0:
                    			tmp = np.random.randint(0,self.n_entity)
                    			if tmp not in self.hr_t[htr[idx, 0]][htr[idx, 2]]:
                        			neg_t = tmp
                        			#print "before ", neg_t_list[idx]
                        			neg_t_list.append(neg_t)
                        			neg_h_list.append(htr[idx,0])
                        		#print "after ", neg_t_list[idx]
            		else:            
                		neg_h = -1
                		while neg_h < 0:
                    			tmp = np.random.randint(0,self.n_entity)
                    			if tmp not in self.tr_h[htr[idx, 1]][htr[idx, 2]]:
                        			neg_h = tmp
                        			neg_h_list.append(neg_h)
                        			neg_t_list.append(htr[idx,1])
        	neg_h_list = np.asarray(neg_h_list,dtype=np.int32)
        	neg_t_list = np.asarray(neg_t_list,dtype=np.int32)                
        	return htr[:,0] , htr[:,2], htr[:, 1], neg_h_list, htr[:,2], neg_t_list	

	def load_triple(self,file_path):
            	with codecs.open(file_path, 'r', encoding='utf-8') as f_triple:
            		triples = []
            		for line in f_triple.readlines():
            			tokens = line.strip().split("\t")
            			if tokens[0] in self.freebase_wikidata and tokens[1] in self.freebase_wikidata:
            				triples.append([self.entity_vocab[self.freebase_wikidata[tokens[0]]], self.entity_vocab[self.freebase_wikidata[tokens[1]]], self.relation_vocab[tokens[2]]])
            	return np.array([np.array(t) for t in triples])

        def gen_hr_t(self,triple_data):
            	hr_t = dict()
            	for h, t, r in triple_data:
                	if h not in hr_t:
                    		hr_t[h] = dict()
                	if r not in hr_t[h]:
                    		hr_t[h][r] = set()
                	hr_t[h][r].add(t)
            	return hr_t

        def gen_tr_h(self,triple_data):
            	tr_h = dict()
            	for h, t, r in triple_data:
                	if t not in tr_h:
                    		tr_h[t] = dict()
                	if r not in tr_h[t]:
                    		tr_h[t][r] = set()
                	tr_h[t][r].add(h)
            	return tr_h

        def stat_dict(self, hr, tr):
        	stat_hr = dict()
        	for key, value in hr.items():
        		for k, val in value.items():
        			if k not in stat_hr:
        				stat_hr[k] = dict()
        				stat_hr[k]['tail'] = len(val)
        				stat_hr[k]['c_h'] = 1
        				stat_hr[k]['head'] = 0
        				stat_hr[k]['c_t'] = 0
        			else:
        				stat_hr[k]['tail'] += len(val)
        				stat_hr[k]['c_h'] += 1
        	for key, value in tr.items():
        		for k, val in value.items():
	        		stat_hr[k]['head'] += len(val)
	        		stat_hr[k]['c_t'] += 1
	        # Key is the rlation id and at index 0 it has average head per tail and at index 1 it has average tail per head	
	        stat_r = dict()	
		for key, value in stat_hr.items():
			stat_r[key] = list()
			stat_r[key].append(float(stat_hr[key]['head'])/stat_hr[key]['c_t'])
			stat_r[key].append(float(stat_hr[key]['tail'])/stat_hr[key]['c_h'])
			#stat_hr[key]['head'] /= stat_hr[key]['c_t']
			#stat_hr[key]['tail'] /= stat_hr[key]['c_h']
			#print self.id_relation_map[key], key, stat_r[key][0], stat_r[key][1], stat_hr[key]['head'], stat_hr[key]['head']
        	return stat_r

        def class_rel(self,stat_r):
        	class_r = dict()
        	for key, val in stat_r.items():
        		#one-one
        		if val[0] <= 1.5 and val[1] <= 1.5:
        			class_r[key] = 0
        		#many-many	
        		elif val[0] > 1.5 and val[1] > 1.5:
        			class_r[key] = 1
        		#one-many	
        		elif val[0] <= 1.5 and val[1] > 1.5:
        			class_r[key] = 2
        		#many-one	
        		elif val[0] > 1.5 and val[1] <= 1.5:			
        			class_r[key] = 3
        	return class_r
        			
	def create_vocab(self):
		# Read Train Data
		with codecs.open(os.path.join(self.data_dir, 'types.json'),'r',encoding='utf-8') as data_file:
			self.types = json.load(data_file)
		print 'Successfully Loaded relevant types'

		with codecs.open(os.path.join(self.data_dir,'freebase_wiki_map.json'),'r',encoding='utf-8') as data_file:
        		self.freebase_wikidata = json.load(data_file)
    		print 'Successfully Loaded Freebase Wikidata mapping'

    		# Create a entity vocab from mapping
    		self.entity_vocab = {self.unk_symbol:self.unk_word_id}
    		for key, value in self.freebase_wikidata.items():
    			self.entity_vocab[value] = len(self.entity_vocab)
    		self.n_entity = len(self.entity_vocab)	

    		# Create a type vocab and ordering of feasbile entity type pair
		self.type_vocab = {} 
		self.lambda_vocab = {}
		for key, value in self.types.items():
			#print key, value	
			self.type_vocab[key] = len(self.type_vocab)
			for v in value:
				self.lambda_vocab[key+v] = len(self.lambda_vocab) + 1

		print "Number of entity type pair ", len(self.lambda_vocab)
		print "Number of Entities", len(self.entity_vocab)
		print "Number of types", len(self.type_vocab)
		# Load realtion vocab
		with codecs.open(os.path.join(self.data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
	            self.relation_vocab = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
        	    self.id_relation_map = {v: k for k, v in self.relation_vocab.items()}
		         			
		print "Number of relations", len(self.relation_vocab)

		self.train_triple = self.load_triple(os.path.join(self.data_dir, 'train.txt'))
	        print("N_TRAIN_TRIPLES: %d" % self.train_triple.shape[0])

	        self.test_triple = self.load_triple(os.path.join(self.data_dir, 'test.txt'))
	        print("N_TEST_TRIPLES: %d" % self.test_triple.shape[0])

	        self.valid_triple = self.load_triple(os.path.join(self.data_dir, 'valid.txt'))
	        print("N_VALID_TRIPLES: %d" % self.valid_triple.shape[0])

	        self.hr_t = self.gen_hr_t(np.concatenate([self.train_triple, self.test_triple, self.valid_triple], axis=0))
        	self.tr_h = self.gen_tr_h(np.concatenate([self.train_triple, self.test_triple, self.valid_triple], axis=0))
        	self.stat_r = self.stat_dict(self.hr_t, self.tr_h)
        	print "Number of relation present in dataset ", len(self.stat_r)
        	self.class_r = self.class_rel(self.stat_r)

	def create_batch_epoch(self, batch_size):
		batch_data = []
		for key, value in self.types.items():
			type_id = [self.type_vocab[key]]
			entity_type_id = []
			entity_id = []
			if len(value) <= self.max_entity:
				entity_type_id = self.pad(self.convert_ent_type_id(key,value),self.max_entity)
				entity_id = self.pad(self.convert_ent_id(value),self.max_entity)
			else:
				ents = np.random.choice(value,self.max_entity,replace =False)	
				entity_type_id = self.convert_ent_type_id(key,ents)
				entity_id = self.convert_ent_id(ents)
			batch_data.append([type_id,entity_type_id,entity_id])

		n_type = len(self.types)
	        start = 0
	        while start < n_type:
	            	end = min(start + batch_size, n_type)
	            	yield batch_data[start:end]
	            	start = end
	
	def get_batch_data(self,batch_data):
		type_ids = np.array(batch_data)[:,0]
		#print type_ids
		type_ids = np.array([xi[0] for xi in type_ids])
		entity_type_ids = np.array(batch_data)[:,1]
		entity_ids = np.array(batch_data)[:,2]
		#print  entity_type_ids.shape
		loss_weights = np.array([np.array([0.0 if xij==self.unk_word_id else 1.0 for xij in xi]) for xi in  entity_ids])
		entity_type_ids = np.array([np.array(xi) for xi in entity_type_ids])
		entity_ids = np.array([np.array(xi) for xi in entity_ids])
		#print type_ids.shape
		#print entity_type_ids.shape
		return type_ids, entity_type_ids, entity_ids, loss_weights


if __name__=="__main__":
	param = json.load(open(sys.argv[1]))
	print param
 	p  = prepareData(param)
 	nbatches_count = 0

 	train_data = []
 	for dat in p.create_batch_epoch(batch_size=param['batch_size']):
		train_data.extend(dat)
		nbatches_count += 1	
	p.get_batch_data(train_data)
	training_data_rel = []
        for dat in p.raw_training_data():
                training_data_rel.append(dat)
              	print dat
	

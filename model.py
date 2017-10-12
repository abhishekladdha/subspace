import tensorflow as tf
import numpy as np
import math
import os
import sys
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.python.framework import dtypes, ops, function
from tensorflow.python.ops import array_ops, nn_ops, math_ops
from tensorflow.python.util import nest

linear = rnn_cell._linear  # pylint: disable=protected-access

def matrix_symmetric(x):
    	return (x + tf.transpose(x, [0,2,1])) / 2

def get_eigen_K(x, square=False):
	if square:
		x = tf.square(x)
	res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
	res += tf.eye(tf.shape(res)[1])
	res = 1 / res
	res -= tf.eye(tf.shape(res)[1])

	# Keep the results clean
	res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
	res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
	return res

#@function.Defun(tf.float32, tf.float32)
@tf.RegisterGradient("Svd")
def gradient_svd(op, grad, grad_u, grad_v):
	#print op
	s, u, v = op.outputs
	v_t = tf.transpose(v, [0,2,1])

	with tf.name_scope('K'):
		K = get_eigen_K(s, True)
	inner = matrix_symmetric(K * tf.matmul(v_t, grad_v))

		# Create the shape accordingly.
	u_shape = u.get_shape()[1].value
	v_shape = v.get_shape()[1].value

		# Recover the complete S matrices and its gradient
	eye_mat = tf.eye(v_shape, u_shape)
	realS = tf.matmul(tf.reshape(tf.matrix_diag(s), [-1, v_shape]), eye_mat)
	realS = tf.transpose(tf.reshape(realS, [-1, v_shape, u_shape]), [0, 2, 1])

	real_grad_S = tf.matmul(tf.reshape(tf.matrix_diag(grad), [-1, v_shape]), eye_mat)
	real_grad_S = tf.transpose(tf.reshape(real_grad_S, [-1, v_shape, u_shape]), [0, 2, 1])

	dxdz = tf.matmul(u, tf.matmul(2 * tf.matmul(realS, inner) + real_grad_S, v_t))
	return dxdz

# @function.Defun(dtypes.float32, dtypes.float32)
# def nuclear_norm_grad(x, dy):
#     	_, U, V = tf.svd(x, full_matrices=False, compute_uv=True)
#     	grad = tf.matmul(U, tf.transpose(V))
#     	return dy * grad


# @function.Defun(dtypes.float32, grad_func=nuclear_norm_grad)
# def nuclear_norm(x):
# 	with tf.device('/cpu:0'):
#     		sigma, _, _ = tf.svd(x, full_matrices=False, compute_uv=True)
#     		norm = tf.reduce_sum(sigma)
#     	return norm

class TypeSubspace:
	"""docstring for ClassName"""
	def __init__(self, param):
		self.num_steps = param['max_len']
		self.num_entity_dim = param['num_entity_dim']
		self.num_type_dim = param['num_type_dim']
		self.num_types = param['num_types']
		self.num_entity = param['num_entity']
		self.num_entity_type = param['num_entity_type']
		self.num_rel = param['num_rel']
		self.learning_rate = param['learning_rate']
		self.batch_size = param['batch_size']
		self.max_gradient_norm = param['max_gradient_norm']
		self.regularization_weight = param['regularization_weight'] 

		max_val = 6. / np.sqrt((2*(self.num_type_dim)))
		# r = tf.random_uniform([self.num_types,self.num_type_dim,self.num_type_dim],-max_val,max_val)
		# r = tf.matmul(tf.transpose(r,[0,2,1]),r)
		#self.type_embedding = tf.get_variable('type_embedding', initializer=r)
		self.type_embedding = tf.get_variable('type_embedding',shape=[self.num_types, self.num_type_dim, self.num_type_dim], initializer=tf.random_uniform_initializer(-max_val, max_val))
		max_val = 6. / np.sqrt((2*(self.num_entity_dim)))
		self.ent_embedding = tf.get_variable('ent_embedding',shape=[self.num_entity, self.num_entity_dim], initializer=tf.random_uniform_initializer(-max_val, max_val))
		self.lamda = tf.get_variable('lambda',shape=[self.num_entity_type, self.num_type_dim], initializer=tf.truncated_normal_initializer(stddev=1e-2))
		self.rel_embedding = tf.get_variable('rel_embedding',shape=[self.num_rel, self.num_entity_dim], initializer=tf.random_uniform_initializer(-max_val, max_val))

		self.pr = None

	def create_placeholder(self):

		self.types = tf.placeholder(tf.int32,[None], name="input_types")
		self.entities_types = tf.placeholder(tf.int32,[None,self.num_steps], name="entities_types")
		self.entities = tf.placeholder(tf.int32,[None,self.num_steps], name="entities")
		self.loss_weights = tf.placeholder(tf.float32,[None, self.num_steps], name="entity_weight") 

		self.pos_h = tf.placeholder(tf.int32, [None], name="pos_h")
	        self.pos_t = tf.placeholder(tf.int32, [None], name="pos_t")
	        self.pos_r = tf.placeholder(tf.int32, [None], name="pos_r")

	        self.neg_h = tf.placeholder(tf.int32, [None], name="neg_h")
	        self.neg_t = tf.placeholder(tf.int32, [None], name="neg_t")
	        self.neg_r = tf.placeholder(tf.int32, [None], name="neg_r")
	        self.test_input = tf.placeholder(tf.int32, [None, 3])

	def lc_entity_loss(self):
		# batch_size x num_type_dim x num_type_dim
		type_spaces = tf.nn.embedding_lookup(self.type_embedding,self.types)
		# batch_size x num_step x num_type_dim x 1
		lamda_s_e = tf.expand_dims(tf.nn.embedding_lookup(self.lamda, self.entities_types), axis = 3)
		# batch_size  x num_step x num_type_dim
		lc = tf.reduce_sum(tf.multiply(tf.expand_dims(type_spaces,axis=1),lamda_s_e),reduction_indices=[2])
		#print lc.get_shape()
		n_norm = tf.reduce_sum(tf.svd(type_spaces,compute_uv=True)[0])
		#self.pr = tf.Print(n_norm,[n_norm, tf.svd(type_spaces,compute_uv=True)[0]], summarize=100) 
		#n_norm = nuclear_norm(type_spaces)
		subspace_loss = self.loss_weights*tf.reduce_sum((tf.nn.embedding_lookup(self.ent_embedding,self.entities) - lc )**2, reduction_indices=[2])
		subspace_loss = tf.reduce_sum(subspace_loss)

		return subspace_loss, n_norm

	def transe_loss(self):

		pos_h_e = tf.nn.embedding_lookup(self.ent_embedding, self.pos_h)
	        pos_t_e = tf.nn.embedding_lookup(self.ent_embedding, self.pos_t)
	        pos_r_e = tf.nn.embedding_lookup(self.rel_embedding, self.pos_r)
	        neg_h_e = tf.nn.embedding_lookup(self.ent_embedding, self.neg_h)
	        neg_t_e = tf.nn.embedding_lookup(self.ent_embedding, self.neg_t)
	        neg_r_e = tf.nn.embedding_lookup(self.rel_embedding, self.neg_r)
	           
	        pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
	        neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)

	        rel_loss = tf.reduce_sum(tf.maximum(pos - neg + 1, 0))
	        #self.pr = tf.Print(rel_loss,[pos, neg, tf.maximum(pos - neg + 2, 0)],summarize=100) 

	        #regularizer_loss = tf.reduce_sum((self.ent_embedding)**2)  + tf.reduce_sum((self.rel_embedding)**2)
	        regularizer_loss = tf.reduce_sum(tf.maximum(tf.reduce_sum((self.ent_embedding)**2,reduction_indices=[1])-1,0)) + tf.reduce_sum(tf.maximum(tf.reduce_sum((self.rel_embedding)**2,reduction_indices=[1])-1,0)) 
	        #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

	        return rel_loss, regularizer_loss 

	def loss(self):
		#subspace_loss, n_norm = self.lc_entity_loss()
		rel_loss, regularizer_loss = self.transe_loss()
		self.pr = tf.Print(regularizer_loss,[rel_loss, regularizer_loss])
		#self.pr = tf.Print(rel_loss,[subspace_loss, n_norm, rel_loss, regularizer_loss, tf.reduce_sum((self.lamda)**2)]) 
		#return subspace_loss + 0.1*n_norm + rel_loss + .1*regularizer_loss
		return rel_loss + .1*regularizer_loss
			
        def inference(self):
		
		h = tf.nn.embedding_lookup(self.ent_embedding, self.test_input[:, 0])
            	t = tf.nn.embedding_lookup(self.ent_embedding, self.test_input[:, 1])
            	r = tf.nn.embedding_lookup(self.rel_embedding, self.test_input[:, 2])

            	#ent_mat = tf.transpose(self.ent_embedding)
            	ent_mat = self.ent_embedding
            	#trh_res = tf.matmul(proj_h+r, ent_mat)
            	trh_res = -tf.reduce_sum((tf.subtract(tf.expand_dims(h+r,axis=1),tf.expand_dims(ent_mat,axis=0)))**2,2)
            	_, tail_ids = tf.nn.top_k(trh_res, k=self.num_entity)

            	#hrh_res = tf.matmul(proj_t-r, ent_mat)            
            	hrh_res = -tf.reduce_sum((tf.subtract(tf.expand_dims(t-r,axis=1),tf.expand_dims(ent_mat,axis=0)))**2,2)
            	print hrh_res.get_shape()
            	_, head_ids = tf.nn.top_k(hrh_res, k=self.num_entity)

            	return head_ids, tail_ids
    	

	def train(self, losses):
		parameters=tf.trainable_variables()
		self.global_step=tf.Variable(0,name="global_step",trainable='False')
		#self.learning_rate= tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.98)
	        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
	        #optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
	        #gradients = optimizer.compute_gradients(losses)
        	# gradients=tf.gradients(losses,parameters)
        	# clipped_gradients,norm=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
        	train_op=optimizer.minimize(losses,global_step=self.global_step)
        	# train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=self.global_step)
	        # #train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=self.global_step)
        	# return train_op, clipped_gradients
        	return train_op
# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import math
import sys
import os
import pickle as pkl
import random
import numpy as np
import nltk
from model import *
import copy
from data_fb15k import *

unk_symbol_index = 0

def run_training(param,data):

	def test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
    		assert len(testing_data) == len(head_pred)
    		assert len(testing_data) == len(tail_pred)

    		relation_specific_eval = np.full(shape= (len(head_pred),4,4), fill_value = -1)

    		mean_rank_h = list()
    		mean_rank_t = list()
    		filtered_mean_rank_h = list()
    		filtered_mean_rank_t = list()

    		for i in range(len(testing_data)):
        		h = testing_data[i, 0]
        		t = testing_data[i, 1]
        		r = testing_data[i, 2]
        		cla = data.class_r[r]
        		# mean rank
        		#print h, r, t, head_pred[i] 
	        	mr = 0
	        	for val in head_pred[i]:
	            		if val == h:
	                		mean_rank_h.append(mr)
	                		relation_specific_eval[i][cla][0] = mr
	                		break
	            		mr += 1
	        	mr = 0
	        	for val in tail_pred[i]:
	            		if val == t:
	                		mean_rank_t.append(mr)
	                		relation_specific_eval[i][cla][1] = mr
	                		break	
	            		mr += 1
	        	# filtered mean rank
	        	fmr = 0
	        	for val in head_pred[i]:
	            		if val == h:
	                		filtered_mean_rank_h.append(fmr)
	                		relation_specific_eval[i][cla][2] = fmr
	                		break
	            		if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
	                		continue
	            		else:
	                		fmr += 1

	        	fmr = 0
	        	for val in tail_pred[i]:
	            		if val == t:
	                		filtered_mean_rank_t.append(fmr)
	                		relation_specific_eval[i][cla][3] = fmr
	                		break
	            		if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
	                		continue
	            		else:
	                		fmr += 1

    		return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t), relation_specific_eval

    	def print_realtion_eval(m):
    		#m.shape
    		m1 = m[:,0,2]
    		m2 = m[:,0,3]
    		print ("FILTERED HITS@10 HEADS %.3f TAILS %.3f for one-one" % (np.mean(m1[m1 != -1] < 10), np.mean(m2[m2 != -1] < 10)))
    		m1 = m[:,1,2]
    		m2 = m[:,1,3]
    		print ("FILTERED HITS@10 HEADS %.3f TAILS %.3f for many-many" % (np.mean(m1[m1 != -1] < 10), np.mean(m2[m2 != -1] < 10)))	
    		m1 = m[:,2,2]
    		m2 = m[:,2,3]
    		print ("FILTERED HITS@10 HEADS %.3f TAILS %.3f for one-many" % (np.mean(m1[m1 != -1] < 10), np.mean(m2[m2 != -1] < 10)))
    		m1 = m[:,3,2]
    		m2 = m[:,3,3]
    		print ("FILTERED HITS@10 HEADS %.3f TAILS %.3f for many-one" % (np.mean(m1[m1 != -1] < 10), np.mean(m2[m2 != -1] < 10)))		

	def evaluate(model, valid_data, epoch, test_type):
		relation_specific_eval_full = np.expand_dims(np.full(shape=(4,4), fill_value = -1),0)
		print relation_specific_eval_full.shape
		accu_mean_rank_h = list()
                accu_mean_rank_t = list()
                accu_filtered_mean_rank_h = list()
                accu_filtered_mean_rank_t = list()

                evaluation_count = 0
                evaluation_batch = []
                for dat in valid_data:
                        head_pred, tail_pred = sess.run([model_pred_h, model_pred_t],{model.test_input: dat})
                        (mrh, fmrh), (mrt, fmrt), relation_specific_eval = test_evaluation(dat, head_pred, tail_pred,data.hr_t, data.tr_h)
                        #relation_specific_eval_full = np.concatenate((relation_specific_eval_full,relation_specific_eval),axis=0)
                      	#print_realtion_eval(relation_specific_eval_full)
                        evaluation_batch.append((dat, head_pred, tail_pred))
                        evaluation_count += 1

                while evaluation_count > 0:
                        evaluation_count -= 1

                        #(mrh, fmrh), (mrt, fmrt) = result_queue.get()
                        testing_data, head_pred, tail_pred = evaluation_batch[evaluation_count-1]
                        (mrh, fmrh), (mrt, fmrt), relation_specific_eval = test_evaluation(testing_data, head_pred, tail_pred,data.hr_t, data.tr_h)
                        relation_specific_eval_full = np.concatenate((relation_specific_eval_full,relation_specific_eval),axis=0)
                        accu_mean_rank_h += mrh
                        accu_mean_rank_t += mrt
                        accu_filtered_mean_rank_h += fmrh
                        accu_filtered_mean_rank_t += fmrt

                print_realtion_eval(relation_specific_eval_full)        
                print("[%s] ITER %d [HEAD PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %(test_type, epoch, np.mean(accu_mean_rank_h), np.mean(accu_filtered_mean_rank_h),np.mean(np.asarray(accu_mean_rank_h, dtype=np.int32) < 10),np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 10)))
		sys.stdout.flush()

		print("[%s] ITER %d [TAIL PREDICTION] MEAN RANK: %.1f FILTERED MEAN RANK %.1f HIT@10 %.3f FILTERED HIT@10 %.3f" %(test_type, epoch, np.mean(accu_mean_rank_t), np.mean(accu_filtered_mean_rank_t),np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 10),np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 10)))

	def perform_training(model, train_batch_rel, train_batch_type, batch_size, step, show_grad_freq):
		type_ids, entity_type_ids, entity_ids, loss_weights = data.get_batch_data(train_batch_type)

		feed_dict = {model.types:type_ids, model.entities_types: entity_type_ids, model.loss_weights:loss_weights, model.entities:entity_ids , model.pos_h: train_batch_rel[0],model.pos_r: train_batch_rel[1],model.pos_t: train_batch_rel[2],model.neg_h: train_batch_rel[3],model.neg_r: train_batch_rel[4],model.neg_t: train_batch_rel[5]}
		loss_type, _, _ = sess.run([losses, train_op, model.pr], feed_dict=feed_dict)
		loss_type = np.sum(np.array(loss_type))
		#print train_batch_loss_entity
		#sum_loss = np.sum(loss)
		# if step % show_grad_freq == 0:
		# 	print "Showing gradient"
	 #                grad_vals = sess.run(gradients, feed_dict= feed_dict)
  #       	        var_to_grad = {}
  #       	        #print grad_vals
  #       	        #print gradients
  #               	for grad_val, var in zip(grad_vals, gradients):
  #               		#print type(grad_val).__module__
  #                       	#if type(grad_val).__module__ == np.__name__:
  #                               var_to_grad[var.name] = grad_val
	 #                        #sys.stdout.flush()
  #       	                print 'var.name ', var.name, 'shape(grad) ', 'mean(grad) ',np.mean(grad_val[0])
  #       	                print grad_val
  #       	                #print 'var.name ', var.name, 'shape(grad) ',grad_val[0],
  #               	        sys.stdout.flush()
		return loss_type

	def get_type_data():
		train_data_type = []
		for dat in data.create_batch_epoch(batch_size=param['batch_size']):
		        train_data_type.append(dat)
		return train_data_type        
		
	param['num_types'] = len(data.type_vocab)
	param['num_entity'] = len(data.entity_vocab)
	param['num_entity_type'] = len(data.lambda_vocab)
	param['num_rel'] = len(data.relation_vocab)
	model_file = os.path.join(param['model_path'],"best_model")
	with tf.Graph().as_default():
		model = TypeSubspace(param)
		model.create_placeholder()
		losses = model.loss()
		#train_op, gradients = model.train(losses)
		train_op = model.train(losses)
		model_pred_h, model_pred_t = model.inference()
		print 'model created'
		sys.stdout.flush()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		config=tf.ConfigProto()
		config.gpu_options.allow_growth=True
		sess = tf.Session(config=config)
		if os.path.isfile(model_file):
			print 'best model exists .. restoring from that point'
			saver.restore(sess, model_file)
		else:
			print 'initializing fresh variables'
			sess.run(init)
			
		all_var = tf.trainable_variables()
		print 'printing all ', len(all_var), 'TF variables '
		for var in all_var:
			print var.name, var.get_shape()
		print 'training started'

		validation_data_rel = []
            	for dat in data.validation_data():
                	validation_data_rel.append(dat)
                testing_data_rel = []
            	for dat in data.testing_data():
                	testing_data_rel.append(dat)
	
		sys.stdout.flush()
		overall_step_count = 0
		last_overall_avg_train_loss = float("inf")
		#evaluate(model, valid_data, param['batch_size'], type_vocab, 0, 0,ancestors)
		for epoch in range(param['max_epochs']):
			training_data_rel = []
			n_batches_rel = 0
            		for dat in data.raw_training_data():
                		training_data_rel.append(dat)
                		n_batches_rel += 1

			best_valid_loss = float("inf")
			nbatches_count = n_batches_rel
			#random.shuffle(train_data)
			type_count = 0
			train_data_type = get_type_data()
			train_loss = 0
			for i in range(nbatches_count):
				overall_step_count += 1

				if type_count >= len(train_data_type):
					train_data_type = get_type_data()
					type_count = 0

				train_batch_rel = training_data_rel[i]
				train_batch_type = train_data_type[type_count]
				type_count += 1
				train_batch_loss  = perform_training(model, train_batch_rel, train_batch_type, param['batch_size'], overall_step_count, param['show_grad_freq'])

				if overall_step_count%param['show_freq']==0:
					print ('Epoch %d Step %d train loss (avg over batch) = %.6f' %(epoch, i, train_batch_loss ))
				sys.stdout.flush() 
				train_loss += train_batch_loss
				
			overall_avg_train_loss = train_loss/float(nbatches_count)
			#print 'Validation started'
			if (epoch > 0 and epoch %10 == 0):
			#if (epoch >= 0):
				#final_type_embedding = model.type_embedding.eval(session=sess)
				final_ent_embedding = model.ent_embedding.eval(session=sess)
				final_rel_embedding = model.rel_embedding.eval(session=sess)
				#np.save(param['model_path'] + 'type_embedding',final_type_embedding)
				#np.savetxt(param['model_path'] + 'ent_embedding',final_ent_embedding)
				#np.savetxt(param['model_path'] + 'rel_embedding',final_rel_embedding)
				evaluate(model, validation_data_rel, epoch, "valid")
				#evaluate(model, testing_data_rel, epoch, "test")
				#saver.save(sess, model_file,global_step=epoch)
				
			print('Epoch %d training completed, train loss (avg overall) = %.6f' %(epoch, overall_avg_train_loss))
			if last_overall_avg_train_loss is not None and overall_avg_train_loss > last_overall_avg_train_loss:
				diff = overall_avg_train_loss - last_overall_avg_train_loss
				if diff>param['train_loss_incremenet_tolerance']:
					print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])
		                else:
        		                print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])		  				
			last_overall_avg_train_loss = overall_avg_train_loss
	            	sys.stdout.flush()
	        print 'Training over'

def main():
	param = json.load(open(sys.argv[1]))
	print param
	p  = prepareData(param)
	run_training(param,p)

if __name__=="__main__":
    main()
			

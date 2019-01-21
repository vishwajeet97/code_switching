# python RNNLM.py train.txt test.txt 1
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from collections import Counter, OrderedDict
import sys
from datetime import datetime

# Local file paths
# file = "data/SEAME/train.txt"
# file2 = "data/SEAME/dev.txt"
# file3 = "data/SEAME/test.txt"
# file4 = "data/SEAME/segmentation_train.txt"
# file5 = "data/SEAME/segmentation_dev.txt"
# file6 = "data/SEAME/segmentation_test.txt"

# index1 = []
# index2 = []
# index3 = []
# model_init_path = "models/cond-rnn_model-3.ckpt"
# Server file paths
file = "/exp/data/SEAME/train.txt"
file2 = "/exp/data/SEAME/dev.txt"
file3 = "/exp/data/SEAME/test.txt"
file4 = "data/SEAME/segmentation_train.txt"
file5 = "data/SEAME/segmentation_dev.txt"
file6 = "data/SEAME/segmentation_test.txt"

index1 = []
index2 = []
index3 = []
model_init_path = "/home/vishwajeet/exp/src/models/cond-rnn_model-3-1.ckpt"
model_save_path = model_init_path
model_restore_path = model_init_path

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first seen'
	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__,
			OrderedDict(self))
	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)

def make_combined_data(filename, segfilename):

	file = open(filename,'r')
	seg = open(segfilename, 'r')
	data = []
	for index, line in enumerate(file):
		line = line.rstrip()
		seg_line = seg.readline().rstrip().split()
		for index2, word in enumerate(line.split()):
			data.append([word, int(seg_line[index2])])
		data.append(["</s>", 0])
	file.close()
	seg.close()
	return np.array(data)

def make_wordid_map(data, k):
	"""
	k is the number of least frequently occuring words in the training 
	set that will be treated as <unk> so as to facilitate good estimates of <unk> words
	"""

	counter = OrderedCounter([i[0] for i in data])
	common_words = counter.most_common()
	total_words = sum(counter.values())
	
	item_to_id = dict()
	least_word_dict = dict(common_words[:-k-1:-1])
	print("The percentage of words treated as <unk> %f" % ( sum(least_word_dict.values())*100.0/total_words) )
	item_to_id["<unk>"] = len(item_to_id)
	i = 1
	for word in counter:
		if word not in least_word_dict.keys():
			item_to_id[word] = i
			i += 1
		else:
			item_to_id[word] = 0
			
	return item_to_id

def encode(data, wordid_map):

	wordid_list = []
	for word in data:
		if word in wordid_map.keys():
			wordid_list.append(wordid_map[word])
		else:
			wordid_list.append(wordid_map['<unk>'])
	return np.array(wordid_list)

def make_batch(index,data, wordid_map, batch_index, batch_size, num_steps):
	temp_index = [i+batch_index*num_steps for i in index]
	temp_index2 = [i+batch_index*num_steps+1 for i in index]
	total_batch = [i[0] for i in data[temp_index]]
	total_batch = encode(total_batch, wordid_map)
	total_batch_2 = [i[0] for i in data[temp_index2]]
	total_batch_2 = encode(total_batch_2, wordid_map)
	total_batch_3 = [i[1] for i in data[temp_index]]
	total_batch_4 = [i[1] for i in data[temp_index2]]

	batch_x = []
	seg_x = []
	batch_y = []
	seg_y = []

	for i in range(0,batch_size*num_steps,num_steps):
		temp = total_batch[i:i+num_steps]
		temp2 = total_batch_2[i:i+num_steps]
		temp3 = total_batch_3[i:i+num_steps]
		temp4 = total_batch_4[i:i+num_steps]

		batch_x.append(temp)
		batch_y.append(temp2)
		seg_x.append(temp3)
		seg_y.append(temp4)

	return (batch_x,np.array(seg_x)[...,np.newaxis],batch_y,np.array(seg_y)[...,np.newaxis])

def get_batch(index,data,wordid_map ,batch_index, batch_size, num_steps):

	return make_batch(index,data, wordid_map, batch_index, batch_size, num_steps)

def initialize_index(batch_size,num_steps,length):
	t = length//(batch_size*num_steps)
	index = range(batch_size)
	temp = []
	[temp.extend(range(i*t*num_steps,i*t*num_steps+num_steps)) for i in index]
	return temp

batch_size = 128
num_steps = 32
num_hidden_units = 512
rnn_size = num_hidden_units
num_hidden_layers = 2
grad_clip = 5
momentum = 0.95
init_scale = 0.1
learning_rate = 0.001
epoch = 100
unk_word_k = 1400
lambda_weight = 0.8
word_vocab_size = 24635 - unk_word_k + 1 # Total distinct words - the least words not being considered plus <unk>

with tf.device('/gpu:0'):

	input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
	segmentation_data = tf.placeholder(tf.float32, shape=[batch_size, num_steps, 1])
	target = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
	segmentation_target = tf.placeholder(tf.float32	, shape=[batch_size, num_steps, 1])
	weight_segmentation = tf.placeholder(tf.float32)
	phase = tf.placeholder(tf.bool)
	keep_prob = tf.placeholder(tf.float32)

	embedding = tf.get_variable("embedding", [word_vocab_size, rnn_size])
	softmax_w = tf.get_variable("softmax_w", [rnn_size, word_vocab_size+1])
	softmax_b = tf.get_variable("softmax_b", [word_vocab_size+1])

	inputs = tf.nn.embedding_lookup(embedding, input_data)
	inputs1 = tf.transpose(inputs, [1,0,2])
	inputs_ta = tf.TensorArray(dtype=tf.float32, size=num_steps)
	inputs_ta = inputs_ta.unstack(inputs1)

	segmentation_data1 = tf.transpose(segmentation_data, [1,0,2])
	segmentation_data_ta = tf.TensorArray(dtype=tf.float32, size=num_steps)
	segmentation_data_ta = segmentation_data_ta.unstack(segmentation_data1)

	def rnn_cell():
		return tf.contrib.rnn.DropoutWrapper(
			rnn.LSTMCell(num_hidden_units,reuse=False)
			,output_keep_prob=keep_prob
			,variational_recurrent=True
			,dtype=tf.float32)

	cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])
	rnn_initial_state = cells.zero_state(batch_size, tf.float32)
	initial_segmentation = tf.zeros([batch_size, 1])

	def loop_fn(time, cell_output, cell_state, loop_state):
		emit_output = cell_output  # == None for time == 0
		if cell_output is None:  # time == 0
			next_cell_state = rnn_initial_state
			segmentation_predict = initial_segmentation
		else:
			next_cell_state = cell_state
			logit_output = tf.matmul(cell_output,softmax_w) + softmax_b
			segmentation_predict = tf.cast(logit_output[:,word_vocab_size:] > 0, tf.float32)
		
		elements_finished = (time >= num_steps)
		finished = tf.reduce_all(elements_finished)

		next_input = tf.cond(finished, 
							lambda: tf.zeros([batch_size, rnn_size+1], dtype=tf.float32), 
							lambda: tf.cond(phase, 
											lambda: tf.concat([inputs_ta.read(time), segmentation_data_ta.read(time)], 1),
											lambda: tf.concat([inputs_ta.read(time), segmentation_predict], 1)))
		next_loop_state = segmentation_predict
		return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

	outputs_ta, final_state, last_segmentation = tf.nn.raw_rnn(cells, loop_fn)
	
	outputs = tf.reshape(outputs_ta.stack(),[num_steps*batch_size,rnn_size])
	logits_combined = tf.matmul(outputs,softmax_w) + softmax_b
	logits_combined = tf.transpose(tf.reshape(logits_combined, [num_steps, batch_size, word_vocab_size+1]), [1,0,2])
	
	segmentation_output = logits_combined[:,:,word_vocab_size:]
	logits = logits_combined[:,:,:word_vocab_size]

	loss_words = tf.contrib.seq2seq.sequence_loss(logits
											, target
											, tf.ones([batch_size, num_steps]
											, dtype=tf.float32)
											, average_across_timesteps=True
											, average_across_batch=True)

	loss_segmentation = tf.nn.sigmoid_cross_entropy_with_logits(logits=segmentation_output,
																labels=segmentation_target)

	cost_segmentation = tf.reduce_sum(loss_segmentation) / (num_steps*batch_size) 
	# cost_words = tf.reduce_sum(loss_words) / num_steps

	cost = weight_segmentation*cost_segmentation + loss_words

	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)

	# optimizer = tf.train.AdamOptimizer(learning_rate)
	optimizer = tf.train.GradientDescentOptimizer(1.0)
	train_op = optimizer.apply_gradients(zip(grads, tvars))
	
	print ("Network Created")
	initializer = tf.random_uniform_initializer(-init_scale, init_scale)     
	saver = tf.train.Saver()
	
	data = make_combined_data(file, file4)
	dev_data = make_combined_data(file2, file5)
	test_data = make_combined_data(file3, file6)

	wordid_map = make_wordid_map(data, unk_word_k)
	print ("Training Started")
	init = tf.global_variables_initializer()
	index1 = initialize_index(batch_size,num_steps,len(data))
	index2 = initialize_index(batch_size,num_steps,len(dev_data))
	index3 = initialize_index(batch_size,num_steps,len(test_data))

	config=tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		if not os.path.isfile(model_restore_path+".meta"):
			sess.run(init)
			save_path = saver.save(sess, model_init_path)
			print("Model saved in file: %s" % save_path)
		
		tt = 0
		saver.restore(sess,model_restore_path)
		print("The weight for segmentation loss is %f" % lambda_weight)
		while tt < epoch :
			print ("Epoch %d : " % tt)
			step = 0
			total_cost = 0.0
			total_cost_words = 0.0
			state = sess.run(rnn_initial_state)
			while (step+1)*batch_size*num_steps < len(data):
				batch_x, seg_x, batch_y, seg_y  = get_batch(index1,data, wordid_map ,step, batch_size, num_steps)
				state,train_cost,train_cost_words,_ = sess.run([final_state,cost,loss_words,train_op],
											feed_dict = {input_data:batch_x,
														segmentation_data:seg_x,
														target:batch_y,
														segmentation_target:seg_y,
														weight_segmentation:lambda_weight,
														phase:True,
														rnn_initial_state: state,
														keep_prob :0.4})
				total_cost += train_cost
				total_cost_words += train_cost_words
				step += 1
			print ("Training loss: %f, ppl: %f " % (np.exp(total_cost/step),np.exp(total_cost_words/step)))
			step = 0
			# total_cost = 0.0
			total_cost_words = 0.0
			state, segmentation_state = sess.run([rnn_initial_state, initial_segmentation])
			while (step+1)*batch_size*num_steps < len(dev_data):
				batch_x, seg_x, batch_y, seg_y = get_batch(index2,dev_data, wordid_map ,step, batch_size, num_steps)
				state, dev_cost_words, segmentation_state = sess.run([final_state, loss_words, last_segmentation], 
											feed_dict = {input_data:batch_x,
														segmentation_data:seg_x,
														target:batch_y, 
														segmentation_target:seg_y,
														weight_segmentation:lambda_weight,
														phase:False,
														initial_segmentation: segmentation_state,
														rnn_initial_state: state,
														keep_prob : 1.0})
				# total_cost += dev_cost
				total_cost_words += dev_cost_words
				step += 1

			# total_cost = np.exp(total_cost/step)
			total_cost_words = np.exp(total_cost_words/step)
			print("Dev Perplexity %f" % (total_cost_words))
			tt +=1
			save_path = saver.save(sess, model_save_path)
			# print("Checkpoint at " + str(datetime.now()))

		step = 0
		total_cost = 0.0
		state, segmentation_state = sess.run([rnn_initial_state, initial_segmentation])
		while (step+1)*batch_size*num_steps < len(test_data):
			batch_x, _, batch_y, _ = get_batch(index3,test_data, wordid_map ,step, batch_size, num_steps)
			
			seg_x = np.zeros([batch_size, num_steps, 1])
			seg_y = np.zeros([batch_size, num_steps, 1])

			state,test_cost, segmentation_state = sess.run([final_state, loss_words, last_segmentation],
										feed_dict = {input_data:batch_x,
													segmentation_data:seg_x,
													target:batch_y, 
													segmentation_target:seg_y,
													weight_segmentation:lambda_weight,
													phase:False,
													rnn_initial_state: state,
													initial_segmentation: segmentation_state,
													keep_prob : 1.0})
			total_cost += test_cost
			step += 1
		print ("Testing perplexity %f" % np.exp(total_cost/step))

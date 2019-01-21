# python RNNLM.py train.txt test.txt 1
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.python import debug as tf_debug
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
from collections import Counter, OrderedDict
import sys

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
	total_batch_3 = [0.0 for i in data[temp_index]]
	batch_x = []
	seg_x = []
	batch_y = []
	for i in range(0,batch_size*num_steps,num_steps):
		temp = total_batch[i:i+num_steps]
		temp2 = total_batch_2[i:i+num_steps]
		temp3 = total_batch_3[i:i+num_steps]
		batch_x.append(temp)
		batch_y.append(temp2)
		seg_x.append(temp3)
	return (batch_x,np.array(seg_x)[...,np.newaxis],batch_y)

def get_batch(index,data,wordid_map ,batch_index, batch_size, num_steps):

	return make_batch(index,data, wordid_map, batch_index, batch_size, num_steps)

def initialize_index(batch_size,num_steps,length):
	t = length//(batch_size*num_steps)
	index = range(batch_size)
	temp = []
	[temp.extend(range(i*t*num_steps,i*t*num_steps+num_steps)) for i in index]
	return temp

if __name__ == '__main__':
	print(sys.argv)
	file = sys.argv[1]
	file2 = sys.argv[2]
	file3 = sys.argv[3]
	file4 = sys.argv[4]

	index1 = []
	index2 = []
	model_init_path = "models/cond-rnn_model-2.ckpt"
	model_save_path = model_init_path
	model_restore_path = model_init_path

	batch_size = 128
	num_steps = 32
	num_hidden_units = 512
	rnn_size = num_hidden_units
	num_hidden_layers = 1
	grad_clip = 5
	momentum = 0.95
	init_scale = 0.1
	learning_rate = 0.001
	epoch = 0
	unk_word_k = 1400
	epsilon = 1e-10
	word_vocab_size = 24635 - unk_word_k + 1 # Total distinct words - the least words not being considered plus <unk>

	with tf.device('/gpu:0'):

		input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
		segmentation_data = tf.placeholder(tf.float32, shape=[batch_size, num_steps, 1])
		target = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
		keep_prob = tf.placeholder(tf.float32)
		embedding = tf.get_variable("embedding", [word_vocab_size, rnn_size])
		inputs = tf.nn.embedding_lookup(embedding, input_data)
		stacked_inputs = tf.concat([inputs, segmentation_data], 2)
		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				rnn.BasicLSTMCell(num_hidden_units,reuse=False)
				,output_keep_prob=keep_prob
				,variational_recurrent=True
				,dtype=tf.float32)

		cells = rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])
		rnn_initial_state = cells.zero_state(batch_size, dtype=tf.float32)
		print(input_data.get_shape().as_list())
		outputs, final_state = tf.nn.dynamic_rnn(cells,stacked_inputs,initial_state=rnn_initial_state,dtype=tf.float32)
		
		outputs = tf.reshape(tf.concat(outputs,1),[-1,rnn_size])
		softmax_w = tf.get_variable("softmax_w", [rnn_size, word_vocab_size])
		softmax_b = tf.get_variable("softmax_b", [word_vocab_size])

		logits = tf.matmul(outputs,softmax_w) + softmax_b + epsilon
		logits = tf.reshape(logits, [batch_size, num_steps, word_vocab_size])	

		softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits),axis=2, keep_dims=True)
		softmax = softmax + epsilon
		entropy = -tf.reduce_sum(softmax * tf.log(softmax),axis=2)

		print(entropy.get_shape().as_list())

		loss = tf.contrib.seq2seq.sequence_loss(logits
												, target
												, tf.ones([batch_size, num_steps]
												, dtype=tf.float32)
												, average_across_timesteps=True
												, average_across_batch=False)

		cost = tf.reduce_sum(loss) / batch_size

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)

		# optimizer = tf.train.AdamOptimizer(learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(1.0)
		train_op = optimizer.apply_gradients(zip(grads, tvars))
		
		print ("Network Created")
		initializer = tf.random_uniform_initializer(-init_scale, init_scale)     
		saver = tf.train.Saver()
		
		data = make_combined_data(file, file3)
		test_data = make_combined_data(file2, file4)
		wordid_map = make_wordid_map(data, unk_word_k)

		init = tf.global_variables_initializer()
		index = initialize_index(batch_size,num_steps,len(test_data))
		config=tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

			saver.restore(sess,model_restore_path)
			step = 0
			total_cost = 0.0
			state = sess.run(rnn_initial_state)
			while (step+1)*batch_size*num_steps < len(test_data):
				batch_x, seg_x , batch_y  = get_batch(index,test_data, wordid_map ,step, batch_size, num_steps)
				for i in range(0, num_steps):
					seg_x1 = np.copy(seg_x)
					seg_x1[:,i,:] = 1.0

					entropy_batch0, = sess.run([entropy,], 
												feed_dict = {input_data:batch_x,
															segmentation_data:seg_x,
															target:batch_y,
															rnn_initial_state: state,
															keep_prob :1.0})
					entropy_batch1, = sess.run([entropy,], 
												feed_dict = {input_data:batch_x,
															segmentation_data:seg_x1,
															target:batch_y,
															rnn_initial_state: state,
															keep_prob :1.0})
					print(np.any(np.isnan(seg_x)), all(i == 0.0 or i == 1.0 for row in seg_x for i in row))
					for j in range(0, batch_size):
						if entropy_batch1[j][i] > entropy_batch0[j][i]:
							seg_x[j][i][0] = 0.0
						else:
							seg_x[j][i][0] = 1.0
					print(np.any(np.isnan(seg_x)))
					
				state,test_cost, loss_np = sess.run([final_state,cost, loss],
											feed_dict = {input_data:batch_x,
														segmentation_data:seg_x,
														target:batch_y,
														rnn_initial_state: state,
														keep_prob :1.0})
				print_array = np.column_stack((seg_x[:,:,0], loss_np))
				print('\n'.join([' '.join(['{:1}'.format(item) for item in row]) for row in print_array]))
				total_cost += test_cost
				step += 1
			print ("Test perplexity %f" % np.exp(total_cost/step))

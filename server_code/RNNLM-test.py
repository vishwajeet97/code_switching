# python RNNLM.py train.txt test.txt 1
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import collections
def make_combined_data(filename):

	file = open(filename,'r')
	data = []
	for line in file:
		line = line.rstrip()
		for word in line.split():
			data.append(word)
		data.append("</s>")
	file.close()
	return np.array(data)

def make_wordid_map(data, k):
	"""
	k is the number of least frequently occuring words in the training 
	set that will be treated as <unk> so as to facilitate good estimates of <unk> words
	"""

	counter = collections.Counter(data)
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
	total_batch = data[temp_index]
	total_batch = encode(total_batch, wordid_map)
	total_batch_2 = encode(data[temp_index2],wordid_map)
	batch_x = []
	batch_y = []
	for i in range(0,batch_size*num_steps,num_steps):
		temp = total_batch[i:i+num_steps]
		temp2 = total_batch_2[i:i+num_steps]
		batch_x.append(temp)
		batch_y.append(temp2)
	return (batch_x,batch_y)

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
	index1 = []
	index2 = []
	model_init_path = "/home/vishwajeet/exp/src/models/rnn_model-2.ckpt"
	model_save_path = model_init_path
	model_restore_path = model_init_path

	batch_size = 64
	num_steps = 32
	num_hidden_units = 512
	rnn_size = num_hidden_units
	num_hidden_layers = 1
	grad_clip = 5
	momentum = 0.95
	init_scale = 0.1
	learning_rate = 0.001
	epoch = 1
	unk_word_k = 20
	word_vocab_size = 24635 - unk_word_k + 1 # Total distinct words - the least words not being considered plus <unk>
	with tf.device('/gpu:0'):

		input_data = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
		target = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
		keep_prob = tf.placeholder(tf.float32)
		embedding = tf.get_variable("embedding", [word_vocab_size, rnn_size])
		inputs = tf.nn.embedding_lookup(embedding, input_data)
		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				rnn.BasicLSTMCell(num_hidden_units,reuse=False)
				,output_keep_prob=keep_prob
				,variational_recurrent=True
				,dtype=tf.float32)

		cells = rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])
		rnn_initial_state = cells.zero_state(batch_size, dtype=tf.float32)
		print(input_data.get_shape().as_list())
		outputs, final_state = tf.nn.dynamic_rnn(cells,inputs,initial_state=rnn_initial_state,dtype=tf.float32)
		
		outputs = tf.reshape(tf.concat(outputs,1),[-1,rnn_size])
		softmax_w = tf.get_variable("softmax_w", [rnn_size, word_vocab_size])
		softmax_b = tf.get_variable("softmax_b", [word_vocab_size])

		logits = tf.matmul(outputs,softmax_w) + softmax_b
		logits = tf.reshape(logits, [batch_size, num_steps, word_vocab_size])	

		loss = tf.contrib.seq2seq.sequence_loss(logits
												, target
												, tf.ones([batch_size, num_steps]
												, dtype=tf.float32)
												, average_across_timesteps=False
												, average_across_batch=True)

		cost = tf.reduce_sum(loss) / num_steps

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)

		# optimizer = tf.train.AdamOptimizer(learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(1.0)
		train_op = optimizer.apply_gradients(zip(grads, tvars))
		
		print ("Network Created")
		initializer = tf.random_uniform_initializer(-init_scale, init_scale)     
		saver = tf.train.Saver()
		
		data = make_combined_data(file)
		test_data = make_combined_data(file2)
		wordid_map = make_wordid_map(data, unk_word_k)
		print ("Training Started")
		init = tf.global_variables_initializer()
		index1 = initialize_index(batch_size,num_steps,len(data))
		index2 = initialize_index(batch_size,num_steps,len(test_data))

		config=tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
				
			saver.restore(sess,model_restore_path)
			step = 0
			total_cost = 0.0
			state = sess.run(rnn_initial_state)
			while (step+1)*batch_size*num_steps < len(test_data):
				batch_x, batch_y  = get_batch(index2,test_data, wordid_map ,step, batch_size, num_steps)
				state,test_cost = sess.run([final_state,cost], 
											feed_dict = {input_data:batch_x,
														target:batch_y, 
														rnn_initial_state: state,
														keep_prob : 1.0})
				total_cost += test_cost
				step += 1

			total_cost = np.exp(total_cost/step)
			print("Test Perplexity %f" % total_cost)

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

text = open('wiki.test.tokens').read()
print('Text length in number of characters:', len(text))

print('Head of text:', text[:1000])

# Print out characters and sort
char = sorted(list(set(text)))
char_size = len(char)

print('Length of characters:', char_size)
print(char)

char2id = dict((c, i) for i, c in enumerate(char))
id2char = dict((i, c) for i, c in enumerate(char))

# Determine probability of next character
def sample(prediction):
	r = random.uniform(0,1)
	s = 0
	char_id = len(prediction) - 1
	for i in range(len(prediction)):
		s += prediction[i]
		if (s >= r):
			char_id = i
			break

	char_one_hot = np.zeros(shape[char_size])
	char_one_hot[char_id] = 1.0

	return char_one_hot

# Vectoizing data
len_per_section = 50
skip = 2
sections = []
next_chars = []

for i in range(0, len(text) - len_per_section, skip):
	sections.append(text[i: i + len_per_section])
	next_chars.append(text[i + len_per_section])

X = np.zeros((len(sections), len_per_section, char_size))
y = np.zeros((len(sections), char_size))

for i, section in enumerate(sections):
	for j, char in enumerate(section):
		X[i, j, char2id[char]] = 1
	y[i, char2id[next_chars[i]]] = 1

print(y)

# Machine Learning Model
batch_size = 512
max_steps = 70000
log_every = 100
save_every = 6000

n_hidden = 1024
text_start = 'I am thinking that'

checkpoint_dir = 'ckpt'

if tf.gfile.Exists(checkpoint_dir):
	tf.gfile.DeleteRecursively(checkpoint_dir)
tf.gfile.MakeDirs(checkpoint_dir)

print('Training data size:', len(X))
print('Approximate steps per epoch:', int(len(X)/batch_size))

# Model
graph = tf.Graph()

with graph.as_default():
	global_step = tf.Variable(0)

	data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
	labels = tf.placeholder(tf.float32, [batch_size, char_size])

	# Input gate - input and output weights + biases
	w_ii = tf.Variable(tf.truncated_normal([char_size, n_hidden], -0.1, 0.1))
	w_io = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	b_i = tf.Variable(tf.zeros([1, n_hidden]))

	# Forget gate
	w_fi = tf.Variable(tf.truncated_normal([char_size, n_hidden], -0.1, 0.1))
	w_fo = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	b_f = tf.Variable(tf.zeros([1, n_hidden]))

	# Output gate
	w_oi = tf.Variable(tf.truncated_normal([char_size, n_hidden], -0.1, 0.1))
	w_oo = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	b_o = tf.Variable(tf.zeros([1, n_hidden]))

	# Memory cell
	w_ci = tf.Variable(tf.truncated_normal([char_size, n_hidden], -0.1, 0.1))
	w_co = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.1, 0.1))
	b_c = tf.Variable(tf.zeros([1, n_hidden]))

	def lstm_cell(i, o, state):
		input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
		forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
		output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
		memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)

		state = forget_gate * state + input_gate * memory_cell
		output = output_gate * tf.tanh(state)

		return output, state

	output = tf.zeros([batch_size, n_hidden])
	state = tf.zeros([batch_size, n_hidden])

	for i in range(len_per_section):
		output, state = lstm_cell(data[:, i, :], output, state)
		if i == 0:
		    outputs_all_i = output
		    labels_all_i = data[:, i+1, :]
		elif i != len_per_section - 1:
		    outputs_all_i = tf.concat(0, [outputs_all_i, output])
		    labels_all_i = tf.concat(0, [labels_all_i, data[:, i+1, :]])
		else:
		    outputs_all_i = tf.concat(0, [outputs_all_i, output])
		    labels_all_i = tf.concat(0, [labels_all_i, labels])

	# Classifier
	w = tf.Variable(tf.truncated_normal([n_hidden, char_size], -0.1, 1.0))
	b = tf.Variable(tf.truncated_normal([char_size]))

	logits = tf.matmul(outputs_all_i, w) + b

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_all_i))

	optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)

with tf.Session(graph=graph) as sess:
	# tf.global_variables_initializer().run()
	offset = 0
	saver = tf.train.Saver()

	for step in range(max_steps):
		offset = offset % len(X)

		if (offset <= (len(X) - batch_size)):
			batch_data = X[offset: offset + batch_size]
			batch_labels = y[offset: offset + batch_size]
			offset += batch_size

		else:
			to_add = batch_size - (len(X) - offset)
			batch_data = np.concatenate((X[offset: len(X)], X[: to_add]))
			batch_labels = np.concatenate((y[offset: len(X)], y[: to_add]))
			offset = to_add

		# Optimization
		_, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})

		if (step % 10 == 0):
			print('Training loss at step {} in time {}: {}'.format(step, datetime.datetime(), training_loss))

			if (step % save_every == 0):
				saver.save(sess, checkpoint_dir + '/model', global_step=step)

# Testing
test_start = 'I plan to make the world a better place'

with tf.Session(graph=graph) as sess:
	tf.global_variables_initializer().run()
	model = tf.train.latest_checkpoint(checkpoint_dir)
	saver = tf.train.Saver()
	saver.restore(sess, model)

	reset_test_state.run()
	test_generated = test_start

	for i in range(len(test_start) - 1):
		test_X = np.zeros((1, char_size))
		test_X[0, char2id[test_start[i]]] = 1.
		_ = sess.run(test_prediction, feed_dict={test_data: test_X})

	test_X = np.zeros((1, char_size))
	test_X[0, char2id[test_start[-1]]] = 1.

	for i in range(500):
		prediction = test_prediction.eval({test_data: test_X})[0]

		next_one_hot_char = sample(prediction)

		next_char = id2char[np.argmax(next_one_hot_char)]

		test_generated += next_char

		test_X = next_one_hot_char.reshape((1, char_size))

	print(test_generated)

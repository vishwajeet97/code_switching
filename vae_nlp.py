import pickle
import itertools
import numpy as np
from scipy import spatial
from scipy.stats import norm
import nltk.data
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
from nltk. corpus import gutenberg
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from keras.layers import Input, Dense, Lambda, Layer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import metrics

w2v = KeyedVectors.load_word2vec_format('/home/ubuntu/pynb/wiki.en.vec')

def split_into_sent (text):
    strg = ''
    for word in text:
        strg += word
        strg += ' '
    strg_cleaned = strg.lower()
    for x in ['\xd5d','\n','"',"!", '#','$','%','&','(',')','*','+',',','-','/',':',';','<','=','>','?','@','[','^',']','_','`','{','|','}','~','\t']:
        strg_cleaned = strg_cleaned.replace(x, '')
    sentences = sent_tokenize(strg_cleaned)
    return sentences

def vectorize_sentences(sentences):
    vectorized = []
    for sentence in sentences:
        byword = sentence.split()
        concat_vector = []
        for word in byword:
            try:
                concat_vector.append(w2v[word])
            except:
                pass
        vectorized.append(concat_vector)
    return vectorized

data_concat = []

for t in [brown.words(), reuters.words(), gutenberg.words()]:
    text = split_into_sent(t)
    vect = vectorize_sentences(text)
    data = [x for x in vect if len(x) == 10]
    for x in data:
        data_concat.append(list(itertools.chain.from_iterable(x)))

with open ('/home/ubuntu/pynb/wikitokens.pickle', 'rb') as f:
    wiki_tokens = pickle.load(f)
wiki_tokens = vectorize_sentences(wiki_tokens)
wikidata = [x for x in wiki_tokens if len(x) == 10]
for x in wikidata:
    data_concat.append(list(itertools.chain.from_iterable(x)))

data_array = np.array(data_concat)
np.random.shuffle(data_array)

train = data_array[:8000]
test = data_array[8000:10000]

batch_size = 500
original_dim = 3000
latent_dim = 1000
intermediate_dim = 1200
epochs = 200
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
vae.compile(optimizer='rmsprop', loss=[zero_loss])

#checkpoint
cp = [callbacks.ModelCheckpoint(filepath="/home/ubuntu/pynb/model.h5", verbose=1, save_best_only=True)]

#train
vae.fit(train, train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test, test), callbacks=cp)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# some matrix magic
def sent_parse(sentence, mat_shape):
    data_concat = []
    word_vecs = vectorize_sentences(sentence)
    for x in word_vecs:
        data_concat.append(list(itertools.chain.from_iterable(x)))
    zero_matr = np.zeros(mat_shape)
    zero_matr[0] = np.array(data_concat)
    return zero_matr

# input: original dimension sentence vector
# output: text
def print_sentence_with_w2v(sent_vect):
    word_sent = ''
    tocut = sent_vect
    for i in range (int(len(sent_vect)/300)):
        word_sent += w2v.most_similar(positive=[tocut[:300]], topn=1)[0][0]
        word_sent += ' '
        tocut = tocut[300:]
    print(word_sent)


# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec



# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

# input: two written sentences, VAE batch-size, dimension of VAE input
# output: the function embeds the sentences in latent-space, and then prints their generated text representations
# along with the text representations of several points in between them
def sent_2_sent(sent1,sent2, batch, dim):
    a = sent_parse([sent1], (batch,dim))
    b = sent_parse([sent2], (batch,dim))
    encode_a = encoder.predict(a, batch_size = batch)
    encode_b = encoder.predict(b, batch_size = batch)
    test_hom = hom_shortest(encode_a[0], encode_b[0], 5)
    
    for point in test_hom:
        p = generator.predict(np.array([point]))[0]
        print_sentence(p)


print_sentence_with_w2v(train[1])
print_sentence_with_w2v(train[2])

sent_encoded = encoder.predict(np.array(train), batch_size = 500)
sent_decoded = generator.predict(sent_encoded)

test_hom = shortest_homology(sent_encoded[3], sent_encoded[10], 5)
for point in test_hom:
    p = generator.predict(np.array([point]))[0]
    print_sentence_with_w2v(p)

test_hom = shortest_homology(sent_encoded[2], sent_encoded[1500], 20)
for point in test_hom:
    p = generator.predict(np.array([find_similar_encoding(point)]))[0]
    print_sentence_with_w2v(p)
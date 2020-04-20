#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import emoji
print(emoji.emojize("let's make a emoji generator :thumbs_up:"))


# In[15]:


data=pd.read_csv("/home/aashish/Desktop/emogen/Training_set.csv")


# In[22]:


data=data.dropna(axis=1)
data1=data.values
data1


# In[32]:


x_train=data1[:,0]
y_train=data1[:,1]
#1st column consists of sentences and 2nd column with code values of emojis


# In[33]:


data2=pd.read_csv("/home/aashish/Desktop/emogen/Test_set.csv")


# In[35]:


data3=data2.values
x_test=data3[:,0]
y_test=data3[:,1]
maxlen=len(max(x_train,key=len).split(' '))


# **labels to emoji conversion and one hot encoding**

# In[45]:


#dict of emojis
emoji_dictionary = {"0": "\u2764\uFE0F",    
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
#using emoji package for conversion with the codes

def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


# In[50]:


#Y is a numpy array of labels and it returns a numpy array of shape (m,maxlabel+1) where m is the length of Y
#for e.g if max label is 4 then there will be 4 0's and 1 one for encoding
def convert_to_one_hot(Y):
    a=np.zeros((Y.shape[0],np.max(Y)+1)) #initialization
    a[np.arange(Y.shape[0]),Y]=1 # storing 1 at the particular label
    return a


# In[58]:


#converstion to type int before encoding
y_train=y_train.astype(int)
#function call
y_one_hot_train=convert_to_one_hot(y_train)
print(str(y_train[3]))
print(str(y_one_hot_train[3]))
#similarly for test
y_test=y_test.astype(int)
y_one_hot_test=convert_to_one_hot(y_test)
print(str(y_train[13]))
print(str(y_one_hot_train[13]))


# In[62]:


#reads glove vector file and returns words to index, index to words, word to vector map[]
def read_glove_vectors(glove_file):
    #open text file as object f
    with open(glove_file,'r') as f:
        #empty set for words
        words=set()
        #dict for word to vec map
        word_to_vec_map={}
        #for every line in file f
        for line in f:
            line=line.strip().split()
            #in every line the first element is the word
            curr_word=line[0]
            #curr word added in set words
            words.add(curr_word)
            #curr word added to vector map using dict
            word_to_vec_map[curr_word]=np.array(line[1:],dtype=np.float64)
        i=1
        words_to_index={}
        index_to_words={}
        for w in sorted(words):
            words_to_index[w]=i
            index_to_words[i]=w
            i=i+1
        return words_to_index, index_to_words,word_to_vec_map


# In[63]:


word_to_index, index_to_word, word_to_vec_map = read_glove_vectors('/home/aashish/Desktop/emogen/glove.6B.50d.txt')


# In[67]:


word_to_index
index_to_word
word_to_vec_map


# **model for prediction of label from sentence**

# In[69]:


#this func inputs a sentence and returns the average of words vector values of all
#the words in the sentence  
def average(sentence,word_to_vec_map):
    words=[i.lower() for i in sentence.split()]
    total=0
    for i in words:
        total=total+word_to_vec_map[i]
    average=total/len(words)
    return average
average("hi this is my project",word_to_vec_map)


# In[74]:


def softmax(z):    
    # We subtract so that value doesn't becomes infinity.
    e_z = np.exp(z - np.max(z))
    
    return e_z / e_z.sum()


# In[77]:




# Training a Simple Neural Network Model. 
def model(X ,Y , word_to_vec_map, learning_rate = 0.01, num_iterations = 1000):
    """Function Paramters: X: A numpy array of shape(m,1) having sentences
       Y: Numpy vector of labels 
       word_to_vec_map: A dictionary containing word to vector mapping
       learning_rate: for gradient descent
       num_iteration: Number of iterations for Gradient Descent.
       Return: Function returns the updated paramters W and b"""

    # Number of training examples.
    m = Y.shape[0]
    
    # Number of output nodes.
    n_y = 5
    
    # Number of input nodes i.e. length of Glove Vector.
    n_h = 50
    
    # Parameter Initialization using Xavier Technique:
    
    W = np.random.randn(n_y , n_h) / np.sqrt(n_h)
    b = np.zeros((n_y , 1))
    
    # Converting Labels to one-hot Vectors.
    Y = Y.astype(int)
    Y_one_hot = convert_to_one_hot(Y)
    
    # Stochastic Gradient Descent:
    for t in range(num_iterations):
        # Looping over the examples one-by-one.
        for i in range(m):
            avg = average(X[i] , word_to_vec_map)
            avg = avg.reshape(50,1)
            # Forward Propagation.
            Z = np.dot(W, avg) + b
            
            A = softmax(Z)
            
            # Cost For Softmax Function;
            
            cost = -(np.sum(np.multiply(Y_one_hot[i].reshape(5,1), np.log(A))))
            
            # Computing Gradients:
            
            # Derivative of cost w.r.t Z
            dz = A - Y_one_hot[i].reshape(5,1)
            # Derivative of cost w.r.t W . We do outer dot product between dz and avg.
            dw = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            # Derivative of cost w.r.t b
            db = dz
            
            # Stochastic Gradient Descent Update:
            
            W = W - learning_rate * dw
            
            b = b - learning_rate * db
            
        if(t%100 == 0 ):   
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
    
    return  W, b



# In[78]:


W, b = model(x_train, y_train, word_to_vec_map)


# In[79]:


# This function predicts labels for Sentences using only Forward Propagation:
def predict(X , Y , W, b, word_to_vec_map):
    """Function Parameters: X: A numpy array of shape(m,1) having sentences
       Y: Numpy vector of labels 
       W,b: Trained Parameters
       word_to_vec_map: A dictionary containing word to vector mapping
       Return: Function returns Label predictions for all examples in X."""
    # Number of examples to be predicted
    m = X.shape[0]
    # Array of Zeros to store prediction labels
    pred = np.zeros((m,1))
    
    # Looping over the training examples
    for i in range(m):
        avg = average(X[i] , word_to_vec_map)
        # Forward Propagation:
        
        Z = np.dot(W, avg.reshape(50,1)) + b
        A = softmax(Z)
        
        # Saving the prediction label in pred vector.
        pred[i] = np.argmax(A)
    #calculating the accuracy
    print("Accuracy = "  + str(np.mean((pred == Y.reshape(Y.shape[0],1)))))
        
    return pred


# In[82]:


# Calling the function to make prediction on training set.
pred = predict(x_train,y_train, W , b , word_to_vec_map)


# **RNN and LSTM implementation**

# In[84]:


# importing keras package and functionalities.
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


# In[90]:


#**words to indices**
#To Use Sequence Model like LSTM, we need to convert sentences into indices first and then eventually use word-embeddings for training.
#We will do padding with 0 vectors and make all sentences of same length. This is a requirement for training using a Sequence Model using a Batch Gradient Descent Approach.
def sentences_to_indices(X , word_to_index, maxLen):
    """Function Parameters: X: Numpy Array having sentences.
       word_to_index: A dictionary mapping words to index.
       maxLen: Length of Longest Sentence
       Return: X_indices : A numpy array of shape (m,10) having indices for all 10 words in a sentence.
       Index 0 means padded word."""
    
    # Number of training examples.
    m = X.shape[0]
    
    # Initialize a X_indices matrix of shape (m , maxLen).We use maxLen for all sentences in training example to make 
    # them of equal length.
    X_indices = np.zeros((m , maxLen))
    
    # Looping over all the training examples.
    for i in range(m):
        # Splitting a sentence into words.
        # Create a list of words having all words in a sentence in lower case.
        words = [j.lower() for j in X[i].split()]
        
        # Initialize word counter to set index as 0: 
        k = 0 
        # Looping over words in a sentence: 
        for w in words:
            # Setting i,jth index in X_indexes:
            X_indices[i][k] = word_to_index[w]
            # To set index of next word, increase k
            k = k + 1 
            
     
    return X_indices



# In[92]:


#Before building the LSTM Network, we need to create an Embedding layer in Keras which can convert word index into Word Vectors.
#creating keras embedding layer
def pretrained_embedding_layer(word_to_vec_map , word_to_index):
    """Function Parameters: word_to_vec_map : A dictionary mapping word to vectors.
       word_to_index: A dictionary mapping word to index.
       Return: Function returns a Keras embedding layer."""
    
    # Length of vocabulary. + 1 is required by Keras. 
    # Because index for words start from 1 and not 0.
    vocab_len = len(word_to_index) + 1 
    # Dimension of embedding vector. We use 50 because our pretrained Glove vec has length 50
    emb_dim = 50
    
    # Initializing an emb_matrix with Zeros. Every row will correspond to vector for that word.
    emb_matrix = np.zeros((vocab_len , emb_dim))
    
    # Looping over each element of word_to_index and saving vectors row-wise in emb_matrix.
    
    for word,index in word_to_index.items():
        
        emb_matrix[index , :] = word_to_vec_map[word]
        
    
    # Defining Keras Embedding Layer.This should have parmaters as Non-Trainable because we don't want to alter the embedding we are using.
    
    embedding_layer = Embedding(vocab_len , emb_dim, trainable = False)
    
    # Before giving weights to embedding layer, we need to build the layer. 
    
    embedding_layer.build((None,))
    
    # Setting the weights for embedding layer.
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
# Embedding_layer can be indexed using 3 index viz 0, index, vector_index.
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][2][:])


# **lstm emoji model**
# 

# In[93]:


def emoji_lstm_model(input_shape , word_to_vec_map , word_to_index):
    """Function Paramters : input_shape: Shape of the input layer for Keras Model.
       word_to_vec_map : Mapping from words to vectors.
       word_to_index : Mapping from words to index.
       Return : Function returns Keras model. """
    
    # We will input the indices rather than words. Creating the input layer for the Network.
    
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Creating the pre-trained embedding layer by calling the above function.
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map , word_to_index)
    
    # Propagating Input through Embedding layer to get word embedding vectors.
    
    embeddings = embedding_layer(sentence_indices)
    
    # Creating LSTM layer with 128 dimensional hidden state:
    # Also, returned sequence should be batch of sequences.
    
    X = LSTM(128 , return_sequences = True)(embeddings)
    
    # Adding Dropout Layer With Probability 0.5:
    
    X = Dropout(0.5)(X)
    
    # Creating Second LSTM Layer with 128 Dimensional Hidden State. 
    # But now, return_sequences = False because we want output at only the last time step and not all time steps.
    
    X = LSTM(128 , return_sequences = False)(X)
    
    # Adding Dropout Layer With Probability 0.5:
    
    X = Dropout(0.5)(X)
    
    # Creating Dense Layer at last time step returning a vector of size 5 with softmax activation.
    #5 bcoz we have 5 diff type pf emojis 
    X = Dense(5)(X)
    
    X = Activation("softmax")(X)
    
    # Creating Model Instance:
    
    model = Model(inputs = sentence_indices , outputs = X)
    
    return model


# In[94]:


# Calling the function to create model.
model = emoji_lstm_model((10,) , word_to_vec_map , word_to_index)


# In[95]:


model.summary()


# In[99]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(x_train, word_to_index, maxlen)


# In[101]:


#X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
# Fitting the model on the training set.
model.fit(X_train_indices, y_one_hot_train, epochs = 50, batch_size = 32, shuffle=True)


# **got 85.7 % accuracy on test set**

# In[106]:


# Converting Sentences to Indices.
X_test_indices = sentences_to_indices(x_test, word_to_index, maxlen)
# Evaluating model on the Test set
loss, acc = model.evaluate(X_test_indices, y_one_hot_test)

print("Test accuracy = ", acc)


# **taking user input for generating relevant emojis**

# In[122]:


text=input()


# In[123]:


# Converting user input sentence to numpy array. 
x_test = np.array([text])
# Converting words to indices. 
X_test_indices = sentences_to_indices(x_test, word_to_index, maxlen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


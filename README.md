Yelp2018SentimentAnalysis

Insights:

Strippping punctuation seems to increase performance by 1% -> Better word tokenizing (Try Punkt tokenizer trained on text from nltk instead of default)

Class-balancing nukes the accuracy by 5%.



Ranking:

1. 65.8%~~

70 words, no punctuation, no stopwords, FB vectors, CNN+LSTM

Epoch 200+ - 27s - loss: 0.4310 - percentageCorrect: 65.9728 - val_loss: 0.4347 - val_percentageCorrect: 65.7453

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 70, 300)           57183900  
_________________________________________________________________
dropout_1 (Dropout)          (None, 70, 300)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 66, 32)            48032     
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 13, 32)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 70)                28840     
_________________________________________________________________
dense_1 (Dense)              (None, 200)               14200     
_________________________________________________________________
dropout_2 (Dropout)          (None, 200)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 200)               40200     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 201       
=================================================================
Total params: 57,315,373
Trainable params: 57,315,373
Non-trainable params: 0
_________________________________________________________________



Predicting: Stars, Funny, Useful, Cool




Ranking:

1. 66.2% accuracy

80 words, no punctuation, no stopwords, FB vectors, CNN

Epoch 123/200
 - 12s - loss: 0.6433 - percentageCorrect: 66.8893 - val_loss: 0.6942 - val_percentageCorrect: 66.2160
___________________________________________________________________________________________________
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Conv1D(filters=16, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=8, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Conv1D(filters=8, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dense(4))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[percentageCorrect])
___________________________________________________________________________________________________

2. 65.34% Accuracy

80 words, puntuaction, no stopwords FB vectors, CNN

Epoch 87/200
 - 11s - loss: 0.6843 - percentageCorrect: 65.4592 - val_loss: 0.6861 - val_percentageCorrect: 65.3440
 
___________________________________________________________________________________________________
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Conv1D(filters=16, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=8, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Conv1D(filters=8, kernel_size=10, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
SpatialDropout1D(0.5)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dense(4))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[percentageCorrect])

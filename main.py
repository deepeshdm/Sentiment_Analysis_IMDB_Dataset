
# Author @ Deepesh Mhatre

# NOTE : Once the model is trained & saved,change
# the path in load_model() as per your local system.

import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM

print("All Dependencies Installed !")

# --------------------------------------------------------------------

df = pd.read_csv("IMDB Dataset.csv")
df["sentiment"].replace({"positive": 1, "negative": 0}, inplace=True)

x = np.array(df["review"].values)
y = np.array(df["sentiment"].values)

x_filtered = []

for review in x:

    # lowercasing the sentence
    review = review.lower()

    # removing punctuations from sentence
    for i in review:
        punc = '''  !()-[]{};:'"\,<>./?@#$%^&*_~  '''
        if i in punc:
            review = review.replace(i, " ")

    x_filtered.append(review)

print("Data Preparation Stage-1 completed !")

# --------------------------------------------------------------------

# One-Hot Encoding each sentence
vocalbulary_size = 5000
onehot_encoded = [one_hot(review, vocalbulary_size) for review in x_filtered]

# Padding each encoded sentence to have a max_length=500
max_length = 500
x_padded = pad_sequences(onehot_encoded, max_length, padding="post")

x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.2)

print("Data Preparation Stage-2 completed !")

# --------------------------------------------------------------------

model = Sequential()
embeded_vector_size = 35
model.add(Embedding(vocalbulary_size, embeded_vector_size, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

print(model.summary())
print("Model Creation Completed !")


# --------------------------------------------------------------------

# Custom Keras callback to stop training when certain accuracy is achieved.
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
            model_name = ("IMDB_sentiment_analysis_" + str(val_acc))
            model.save(model_name)


# Model converges at 0.87 accuracy with current hyperparameters.
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),
          callbacks=[MyThresholdCallback(threshold=0.87)])

# model.save("IMDB_sentiment_analysis")

print("Model Training Completed !")

# --------------------------------------------------------------------

from keras.models import load_model

trained_model = load_model("/content/IMDB_sentiment_analysis_0.8787999749183655")
predicted = trained_model.predict(x_test)[2]

sentiment = 1 if predicted > 0.6 else 0

print("PREDICTED : ", sentiment)
print("ACTUAL : ", y_test[2])


# --------------------------------------------------------------------

# A high-level function to implement everything at once.

def get_sentiment(sentence: str):
    if isinstance(sentence, (str)):
        pass
    else:
        raise Exception("Input needs to be of type 'str' ")

    # filtering the sentence
    sentence = sentence.lower()

    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

    for word in sentence:
        if word in punc:
            sentence = sentence.replace(word, " ")

    # Loading the saved trained model.
    from keras.models import load_model

    trained_model = load_model("/content/IMDB_sentiment_analysis_0.8787999749183655")

    predicted = trained_model.predict(x_test)[2]
    sentiment = 1 if predicted > 0.6 else 0

    if sentiment == 1:
        print("Positive")
    else:
        print("Negative")

    return sentiment


# --------------------------------------------------------------------

get_sentiment("That movie was really good!")

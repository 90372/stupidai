import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
import tensorflow_datasets as tfds

# Load and prepare text training data
dataset, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_data, test_data = dataset["train"], dataset["test"]

# Preprocess the text (remove HTML tags)
def preprocess(text, label):
    text = tf.strings.regex_replace(text, "<br />", " ")
    return text, label

train_data = train_data.map(preprocess).cache().shuffle(10000).batch(32)
test_data = test_data.map(preprocess).batch(32)

# Tokenize the text
vectorize_layer = layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=250)
train_texts = train_data.map(lambda x, y: x).unbatch()
vectorize_layer.adapt(train_texts)

def vectorize_text(text, label):
    return vectorize_layer(text), label

# Apply tokenization before batching
train_data = train_data.map(vectorize_text).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(vectorize_text).batch(32).prefetch(tf.data.AUTOTUNE)

# Language model architecture (LSTM)
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=250),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(10000, activation='softmax')  # Output layer with the size of the vocabulary
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=test_data)

# Save the model for later use
model.save("language_model.h5")

# Flask API for interaction
app = Flask(__name__)

# Load model globally once at startup
model = keras.models.load_model("language_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('text', "")  # Use .get() to prevent KeyError
    if not data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Vectorize the input text (wrap input in a list to pass to the vectorizer)
    vectorized_text = vectorize_layer([data])
    prediction = model.predict(vectorized_text)
    
    # Get the most probable next word
    next_word_index = tf.argmax(prediction[0], axis=-1).numpy()
    
    # Adjust to exclude special tokens
    vocab = vectorize_layer.get_vocabulary()
    
    # If the index corresponds to a special token (e.g., padding or unknown), handle it
    if next_word_index < len(vocab):
        next_word = vocab[next_word_index]
    else:
        next_word = "Unknown"
    
    return jsonify({'prediction': next_word})

if __name__ == '__main__':
    from waitress import serve  # Use Waitress for production stability
    serve(app, host='0.0.0.0', port=5050)  # Change port to avoid conflicts

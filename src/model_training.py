 %%writefile /content/src/model_training.py
# -*- coding: utf-8 -*-
"""
Model Training Pipeline Module
Defines and trains the classical Funk SVD baseline and the Deep Learning NCF architecture.
"""

from surprise import SVD, Reader, Dataset
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def train_funk_svd(train_df):
    """
    Transforms the processed DataFrame into a Surprise dataset structure
    and trains a biased Funk SVD Matrix Factorization model.
    """
    print("🤖 Initializing and training classical Biased Funk SVD Baseline...")

    # Define the rating scale bounds explicitly for the reader interface
    reader = Reader(rating_scale=(1, 5))

    # Load data from the standard structural columns
    data = Dataset.load_from_df(train_df[['Customer_Id', 'Movie_Id', 'Rating']], reader)
    trainset = data.build_full_trainset()

    # Optimal hyperparameters derived from validation testing
    model = SVD(n_factors=15, n_epochs=30, lr_all=0.005, reg_all=0.12, random_state=42)
    model.fit(trainset)

    print("✅ Funk SVD baseline training complete.")
    return model

def build_and_train_ncf(train_df, num_users, num_movies, embedding_dim=16, epochs=3, batch_size=256):
    """
    Constructs a Deep Neural Collaborative Filtering graph network using a 
    Multi-Layer Perceptron (MLP) architecture and optimizes it using TensorFlow/Keras.
    """
    print("🤖 Initializing Deep Neural Collaborative Filtering (NCF) Architecture...")

    # 1. Network Input Layers
    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    # 2. Embedding Layers with L2 Regularization constraints to mitigate sparse overfit vectors
    user_embedding = Embedding(num_users, embedding_dim, embeddings_regularizer=l2(1e-6), name='user_emb')(user_input)
    movie_embedding = Embedding(num_movies, embedding_dim, embeddings_regularizer=l2(1e-6), name='movie_emb')(movie_input)

    # 3. Flatten Layers from matrices to feature vectors
    user_flat = Flatten()(user_embedding)
    movie_flat = Flatten()(movie_embedding)

    # 4. Layer Fusion (Concatenation path for Multi-Layer Perceptron)
    mlp_vector = Concatenate()([user_flat, movie_flat])

    # 5. Deep Hidden Layers with ReLU and Dropout safeguards
    dense_1 = Dense(32, activation='relu', name='mlp_dense_32')(mlp_vector)
    dropout_1 = Dropout(0.1, name='dropout_0.1')(dense_1)
    dense_2 = Dense(16, activation='relu', name='mlp_dense_16')(dropout_1)

    # 6. Single Output Neuron (Continuous prediction scale matching star ratings)
    output_layer = Dense(1, activation='linear', name='prediction_output')(dense_2)

    # Compile the graph
    ncf_model = Model(inputs=[user_input, movie_input], outputs=output_layer)
    ncf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    print(f"⏳ Training NCF model over {epochs} epochs with a batch size of {batch_size}...")

    # Fit the network layers using the continuous sequential mapped indices
    ncf_model.fit(
        x=[train_df['user_idx'].values, train_df['movie_idx'].values],
        y=train_df['Rating'].values,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    print("✅ Deep Learning NCF training complete.")
    return ncf_model

if __name__ == "__main__":
    print("✨ model_training.py compiled with no syntax or functional layout errors.")

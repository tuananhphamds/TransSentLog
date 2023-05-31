import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Masking
from tensorflow.keras.models import load_model, Model, Sequential

HOME = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]

def interpolate_texts(baseline, text, m_steps):
    # Generate m_steps intervals for integral_approximation()
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    alphas_x = alphas[tf.newaxis, :, tf.newaxis, tf.newaxis]
    delta = text - baseline
    texts = baseline + alphas_x * delta
    return texts

def compute_gradients(new_model, t, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(t)
        probs = new_model(t)[:, target_class_idx]
        grads = tape.gradient(probs, t)

    return grads

def batch_interpretation(x_batch, target_label, embedding_layer, new_model, n_steps):
    x_batch = tf.constant(x_batch)
    x_batch_embedded = embedding_layer(x_batch)
    x_batch_embedded_expands = tf.expand_dims(x_batch_embedded, axis=1)
    x_batch_baseline = tf.zeros(shape=tf.shape(x_batch_embedded_expands))

    inter_texts = interpolate_texts(x_batch_baseline, x_batch_embedded_expands, n_steps)

    cur_shape = tf.shape(inter_texts)
    grad_shape = [-1, cur_shape[-2], cur_shape[-1]]
    batch_path_gradients = compute_gradients(new_model, tf.reshape(inter_texts, grad_shape), target_label)
    all_grads = tf.reduce_sum(tf.reshape(batch_path_gradients, cur_shape), axis=1) / n_steps
    x_grads = tf.math.multiply(all_grads, x_batch_embedded)
    weights = tf.reduce_sum(x_grads, axis=-1).numpy()
    return weights

def get_weights(model, x_test, y_pred, n_steps):
    embedding_layer = model.layers[0]
    TARGET_LABEL = 0 # NEGATIVE - ANOMALY

    new_model = Sequential()
    new_model.add(Masking())
    for layer in model.layers[1:]:
        new_model.add(layer)

    idxs = []
    neg_x_pad = []
    for idx, x in enumerate(x_test):
        if y_pred[idx] == TARGET_LABEL:
            neg_x_pad.append(x)
            idxs.append(idx)

    if len(neg_x_pad) == 0:
        return [], []

    # ----------------------------INTERPRET MODEL PREDICTIONS---------------------------
    batch_size = 64
    count = 0
    total_weights = []
    for i in range(int(np.ceil(len(neg_x_pad) / batch_size))):
        x_batch = neg_x_pad[i*batch_size:(i+1)*batch_size]
        weights = batch_interpretation(x_batch, TARGET_LABEL, embedding_layer, new_model, n_steps)
        total_weights.append(weights)
        count += len(weights)

    # -----------------------------SCALE WEIGHTS TO RANGE (0, 1)------------------------
    total_weights = np.concatenate(total_weights, axis=0)
    mins = np.min(total_weights, axis=-1).reshape(-1, 1)
    maxs = np.max(total_weights, axis=-1).reshape(-1, 1)
    scaled_weights = (total_weights - mins) / (maxs - mins)

    return idxs, scaled_weights

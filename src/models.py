import tensorflow as tf
import keras
from tensorflow.keras import layers, Model

@keras.saving.register_keras_serializable(package="MultiClinAI")
def _pad_mask_fn(t):
    return tf.not_equal(t, 0)

@keras.saving.register_keras_serializable(package="MultiClinAI")
def _pad_mask_shape(s):
    return s

@keras.saving.register_keras_serializable(package="MultiClinAI")
def _attn_mask_fn(m):
    m_cast = tf.cast(m, tf.int32)
    return m_cast[:, :, None] * m_cast[:, None, :]

@keras.saving.register_keras_serializable(package="MultiClinAI")
def _attn_mask_shape(s):
    return (s[0], s[1], s[1])

def _mask(word_in):
    return layers.Lambda(_pad_mask_fn, output_shape=_pad_mask_shape, name="pad_mask")(word_in)

def _attn_mask(mask):
    # (batch, T) bool  ->  (batch, T, T) so padding is never attended to
    return layers.Lambda(_attn_mask_fn, output_shape=_attn_mask_shape, name="attn_mask")(mask)

def _rnn(kind, units):
    cell = layers.LSTM if kind == "lstm" else layers.GRU
    return layers.Bidirectional(cell(units, return_sequences=True))

def _char_cnn(char_in, n_chars, char_dim=30, filters=50):
    c = layers.Embedding(n_chars, char_dim)(char_in)                       # (B,T,C,30)
    c = layers.TimeDistributed(layers.Conv1D(filters, 3, padding="same", activation="relu"))(c)
    return layers.TimeDistributed(layers.GlobalMaxPooling1D())(c)         # (B,T,50)

def build_model(name, vocab_size, n_chars, n_labels, max_len, max_chars=20,
                emb=128, units=128, dropout=0.3, heads=4):
    use_chars = name.startswith("cnn")
    use_attn = name.endswith("attn")
    rnn_kind = "gru" if "gru" in name else "lstm"

    word_in = layers.Input(shape=(max_len,), dtype="int32", name="words")
    inputs = [word_in]
    mask = _mask(word_in)

    x = layers.Embedding(vocab_size, emb)(word_in)
    if use_chars:
        char_in = layers.Input(shape=(max_len, max_chars), dtype="int32", name="chars")
        inputs.append(char_in)
        x = layers.Concatenate()([x, _char_cnn(char_in, n_chars)])
        x = layers.Conv1D(emb, 3, padding="same", activation="relu")(x)     # local n-gram context
    x = layers.Dropout(dropout)(x)

    x = _rnn(rnn_kind, units)(x, mask=mask)                                 # (B,T,2*units)

    if use_attn:
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=2 * units // heads)(
            x, x, attention_mask=_attn_mask(mask))
        x = layers.LayerNormalization()(layers.Add()([x, a]))               # residual + norm
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_labels, activation="softmax")(x)
    return Model(inputs, out, name=name)

MODEL_NAMES = ["bilstm", "bilstm_attn", "cnn_bilstm", "cnn_bilstm_attn", "cnn_bigru_attn"]

import tensorflow as tf

src = "mp_hand_gesture"  # folder with saved_model.pb

print("Loading SavedModel from:", src)
model = tf.keras.models.load_model(src)
print("Model loaded.")

# Save as H5 (legacy format, widely supported)
h5_path = "mp_hand_gesture.h5"
model.save(h5_path)
print("Saved H5 model to:", h5_path)

# Save as Keras v3 format (modern, preferred by Keras 3)
keras_path = "mp_hand_gesture.keras"
model.save(keras_path)
print("Saved Keras v3 model to:", keras_path)

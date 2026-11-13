from tensorflow import keras
from tensorflow.keras import layers

def build_feature_extractor(input_shape=(128, 128, 3), num_classes=3):
    """Build a compact CNN; the penultimate 256-unit layer is the feature vector."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    features = layers.Dense(256, activation="relu", name="image_features")(x)
    x = layers.Dropout(0.3)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    classifier = keras.Model(inputs, outputs, name="disaster_cnn")
    extractor = keras.Model(inputs, features, name="cnn_feature_extractor")
    classifier.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return classifier, extractor

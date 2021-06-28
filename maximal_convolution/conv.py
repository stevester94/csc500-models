#! /usr/bin/python3

import sys, os

from tensorflow.python.ops.gen_math_ops import floor

from steves_utils.graphing import save_confusion_matrix, save_loss_curve
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory

from steves_utils import utils


import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import json
import sys
import time

if __name__ == "__main__":
    j = json.loads(sys.stdin.read())

    EXPERIMENT_NAME = j["experiment_name"]
    SOURCE_DATASET_PATH = j["source_dataset_path"]
    LEARNING_RATE = j["learning_rate"]
    ORIGINAL_BATCH_SIZE = j["original_batch_size"]
    DESIRED_BATCH_SIZE = j["desired_batch_size"]
    EPOCHS = j["epochs"]
    PATIENCE = j["patience"]

# EXPERIMENT_NAME = "demo-that-CIDA-is-flawed"
# SOURCE_DATASET_PATH = utils.get_datasets_base_path() + "/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-2.8.14.20.26.32/"
# LEARNING_RATE = 0.0001
# ORIGINAL_BATCH_SIZE = 100
# # DESIRED_BATCH_SIZE = 128
# DESIRED_BATCH_SIZE = 1024
# EPOCHS = 3
# PATIENCE = 10

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def apply_dataset_pipeline(datasets):
    """
    Apply the appropriate dataset pipeline to the datasets returned from the Windowed_Shuffled_Dataset_Factory
    """
    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    train_ds = train_ds.map(
        lambda x: ((x["IQ"], x["distance_feet"]),tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    val_ds = val_ds.map(
        lambda x: ((x["IQ"], x["distance_feet"]),tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )
    test_ds = test_ds.map(
        lambda x: ((x["IQ"], x["distance_feet"]),tf.one_hot(x["serial_number_id"], RANGE)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )


    train_ds = train_ds.unbatch()
    val_ds = val_ds.unbatch()
    test_ds = test_ds.unbatch()

    train_ds = train_ds.shuffle(100 * ORIGINAL_BATCH_SIZE, reshuffle_each_iteration=True)
    
    train_ds = train_ds.batch(DESIRED_BATCH_SIZE)
    val_ds  = val_ds.batch(DESIRED_BATCH_SIZE)
    test_ds = test_ds.batch(DESIRED_BATCH_SIZE)

    train_ds = train_ds.prefetch(100)
    val_ds   = val_ds.prefetch(100)
    test_ds  = test_ds.prefetch(100)

    return train_ds, val_ds, test_ds




def get_shuffled_and_windowed_from_pregen_ds():
    path = SOURCE_DATASET_PATH
    print(path)
    datasets = Windowed_Shuffled_Dataset_Factory(path)

    return apply_dataset_pipeline(datasets)



if __name__ == "__main__":
    start_time = time.time()

    # Hyper Parameters
    RANGE   = len(ALL_SERIAL_NUMBERS)
    DROPOUT = 0.5 # [0,1], the chance to drop an input
    set_seeds(1337)

    train_ds, val_ds, test_ds = get_shuffled_and_windowed_from_pregen_ds()

    input_x  = keras.Input(shape=(2,ORIGINAL_PAPER_SAMPLES_PER_CHUNK))
    input_t  = keras.Input(shape=())

    t = keras.layers.Reshape((1,))(input_t)

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=1,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_conv1d_1"
    )(input_x)

    x = keras.layers.Convolution1D(
        filters=50,
        kernel_size=7,
        strides=2,
        activation="relu",
        kernel_initializer='glorot_uniform',
        data_format="channels_first",
        name="classifier_conv1d_2"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    x = keras.layers.Flatten(name="classifier_flatten")(x)

    # x = keras.layers.Concatenate(name="classifier_concat_x_with_t")([x, t])

    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_fc_1"
    )(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_fc_2"
    )(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_fc_3"
    )(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_fc_4"
    )(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer='he_normal',
            name="classifier_fc_5"
    )(x)
    x = keras.layers.Dropout(DROPOUT)(x)


    x = keras.layers.Dense(
        units=80,
        activation='relu',
        kernel_initializer='he_normal',
        name="classifier_fc_6"
    )(x)

    x = keras.layers.Dropout(DROPOUT)(x)

    outputs = keras.layers.Dense(RANGE, activation="softmax")(x)

    model = keras.Model(inputs=[input_x, input_t], outputs=outputs, name="steves_model")
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()], # Categorical is needed for one hot encoded data
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="./best_weights/weights.ckpt",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            save_weights_only=True,
            monitor="val_loss", # We could theoretically monitor the val loss as well (eh)
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE),
        tf.keras.callbacks.CSVLogger(
            "training_log.csv", separator=',', append=False
        )
    ]

    history = model.fit(
        x=train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    results_csv_path="./results.csv"

    with open("./details.txt", "w") as f:
        f.write("experiment_name {}\n".format(EXPERIMENT_NAME))
        f.write("epochs_trained {}\n".format(len(history.history["loss"])))

    with open(results_csv_path, "w") as f:
        f.write("distance,val_loss,val_accuracy,test_loss,test_accuracy\n")
        

    print("Loading best weights...")
    model.load_weights("./best_weights/weights.ckpt")

    print("Analyze the model...")
    for distance in ALL_DISTANCES_FEET + ["2.8.14.20.26.32"]:
        print("Distance", distance)

        # Distance 4 would not generate a windowed dataset for some reason
        if distance == 4:
            continue

        target_dataset_path = "{datasets_base_path}/automated_windower/windowed_EachDevice-200k_batch-100_stride-20_distances-{distance}".format(
            datasets_base_path=utils.get_datasets_base_path(), distance=distance
        )

        datasets = Windowed_Shuffled_Dataset_Factory(target_dataset_path)

        train_ds, val_ds, test_ds = apply_dataset_pipeline(datasets)

        # Analyze on the test data
        test_results = model.evaluate(
            test_ds,
            verbose=1,
        )

        # Analyze on the val data
        val_results = model.evaluate(
            val_ds,
            verbose=1,
        )

        with open(results_csv_path, "a") as f:
            f.write("{},{},{},{},{}\n".format(distance, val_results[0], val_results[1], test_results[0], test_results[1]))

        print("Calculate the confusion matrix")
        total_confusion = None
        f = None
        for e in test_ds.unbatch().batch(50000).prefetch(5):
            confusion = tf.math.confusion_matrix(
                np.argmax(e[1].numpy(), axis=1),
                np.argmax(model.predict(e[0]), axis=1),
                num_classes=RANGE
            )

            if total_confusion == None:
                total_confusion = confusion
            else:
                total_confusion = total_confusion + confusion
        if distance == "2.8.14.20.26.32":
            save_confusion_matrix(confusion, path="confusion_distance-2_8_14_20_26_32.png")
        else:
            save_confusion_matrix(confusion, path="confusion_distance-{}".format(distance))

    save_loss_curve(history)

    end_time = time.time()

    with open(results_csv_path, "a") as f:
        f.write("total time seconds: {}\n".format(end_time-start_time))
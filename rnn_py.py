import tensorflow as tf
import keras.api._v2.keras as keras
from keras.api._v2.keras import layers, activations, optimizers

def run():
    print('building model...')
    model = keras.Sequential()
    model.add(layers.LSTM(
        64,
        input_shape=(None, 28),
        activation=activations.tanh,
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10))
    print('Summary:')
    model.summary()

    print('Getting MDIST data...')
    mnist = keras.datasets.mnist
    (x_train,   y_train), (x_test, y_test)    = mnist.load_data()
    x_train,    x_test                        = x_train/255.0, x_test/255.0
    x_val,      y_val                         = x_test[:-30], y_test[:-30]
    x_test,     y_test                        = x_test[-30:], y_test[-30:]

    print('Compile model...')
    model.compile(
        loss        = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer   = optimizers.SGD(learning_rate=0.001),
        metrics     = ["accuracy"],
    )

    print('Training and Fitting...')
    model.fit(
        x_train, 
        y_train, 
        validation_data = (x_val, y_val), 
        batch_size      = 32, 
        epochs          = 50,
        callbacks       = [keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)],
    )

    print('Testing...')
    results = []
    for i in range(30):
        result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis = 1)
        results.append(1 if result.numpy() - y_test[i] == 0 else 0)

    print('Final testing accuracy: ' + str(round(sum(results)/len(results), 2)))

if __name__:
    run()
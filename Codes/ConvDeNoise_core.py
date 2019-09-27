import time
import keras
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
        
        
def ConvDeNoise(x_train, xref_train, x_test, xref_test, output_f='./ConvDeNoise.h5' , F_nb=40, K_sz=130, pool=2, patience_N=25, epoch_nb=200, btch_sz=256):
    """
        ConvDeNoise: Main function
        - Inputs:
            - Data: !!!!!! Amplitudes of the SC functions need to be normalized between -1 and 1 !!!!!!
                - x_train: Noisy SC functions of the training set, Shape: (nb_of_waveforms (80% of 363 days of SC functions every 20-min: 20904), length_of_signal (1s at 200 Hz), nb_of_channels (Z-E and Z-N SC functions)) -> (20904, 200, 2)
                - xref_train: Clean SC functions of the training set, Shape: same as 'x_train'
                - x_test: Noisy SC functions of the validation set; Shape: (nb_of_waveforms (20% of 363 days of SC functions every 20-min: 5232), length_of_signal (1s at 200 Hz), nb_of_channels (Z-E and Z-N SC functions)) -> (5232, 200, 2)
                - xref_test: Clean SC functions of the validation set, Shape: same as 'x_test'
            - 1D convolution operation:
                - F_nb: filters, dimension of the output space, i.e.,  number of output filters in the convolution (Integer, default: 40)
                - K_sz: Length of the 1D convolution window (Integer, default: 130):  This will only affect the size of the first and last two 1D convolution operations
            - MaxPooling1D and UpSampling1D operations:
                - pool: size of the max pooling windows and Upsampling factor. (Integer, default: 2)
            - Fit:
                - patience_N: Number of epochs with no improvement of the validation set loss(val_loss) after which training will be stopped. (Integer, default: 25) 
                - epoch_nb: Number of epochs to train the model (Integer, default: 200)
                - btch_sz: Number of samples per gradient update (Integer, default: 256)
        - Output:
            - output_f: HDF5 file with the trained model.
    """
    # Input shape
    input_img = Input(shape=(int(x_train.shape[1]),int(x_train.shape[2])) ) 
    
    # Encoder: wo 1D convolution operations with ReLU activation functions and Max-pooling operations.
    x = Conv1D(int(F_nb), int(K_sz), activation = 'relu', padding='same')(input_img)
    x = MaxPooling1D(pool, padding='same')(x)
    x = Conv1D(int(F_nb), int(K_sz/2), activation='relu', padding='same')(x)
    x = MaxPooling1D(pool, padding='same')(x)

    #Bottelneck
            
    # Decoder
    x = Conv1D(int(F_nb), int(20),activation ='relu', padding='same')(x)
    x = UpSampling1D(pool)(x)
    x = Conv1D(int(F_nb), int(K_sz/2),activation ='relu', padding='same')(x)
    x = UpSampling1D(pool)(x)
    # Output layer
    decoded = Conv1D(int(2), K_sz, activation='tanh', padding='same')(x)
    
    #Set the model that includes all layers required in the computation of "decoded" given "input_img".
    autoencoder = Model(input_img, decoded)
    
    # Set the loss function (or objective function): mean_squared_error with the ADAM optimizer
    autoencoder.compile(optimizer='adam', loss='mean_squared_error' ) 
    
    # Print a summary representation of the model
    autoencoder.summary()
    
    ## Callbacks
    # Set the early stopping by monitoring the loss of the validation set. Training stops is val_loss does not improve for "patie_N" epochs (default, patience_N=20)
    es = EarlyStopping(monitor='val_loss', verbose=0, patience=patience_N) 
    # Save the model after every epoch is val_loss is better (lower)
    mc = ModelCheckpoint(output_f, monitor='val_loss', verbose=0, save_best_only=True)
    # Time the duration of the training
    time_callback = TimeHistory()
    
    ## Fit
    # Train the model for a given number of epochs (epoch_nb) using a batched gradient descent (btch_sz). 
    # The callbacks are used to get a view on internal states and statistics of the model during training.
    history = autoencoder.fit(x_train, xref_train,
                epochs=epoch_nb,
                batch_size=btch_sz,
                shuffle=True,
                validation_data=(x_test, xref_test),
                callbacks=[es, mc, time_callback])
    
    return autoencoder

# ConvDeNoise: A convolutional denoising autoencoder to denoise correlation functions

* Codes to reproduce Figure 7 (Figure 8 can also be plotted by changing two lines of the code) of the following paper:
  - Viens L. and Van Houtte C., Denoising ambient seismic field correlation functions with convolutional autoencoders (submitted to GJI). A preprint will be available soon.
  
* The autoencoder is composed of an encoder part and a decoder part:
![Image of ConvDeNoise](https://github.com/lviens/ConvDeNoise/blob/master/ConvDeNoise_image.png)


* The **Codes** folder contains 5 files: 
  - The **Reproduce_Fig_7.py** file is the python code to reproduce the figure.
  - The **functions_for_autoencoders.py** file contains functions to bandpass filter the data with a Butterworth filter, denoise the SC functions with the SVDWF method (Moreau et al., 2017), and compute the stretching to retrieve dv/v measurements
  - The **ConvDeNoise_NS7M_station.h5** contains the weights of ConvDeNoise trained for the NS7M station (Requires Keras 2.2.4)
  - The **Test_data.mat** contains 16 days of raw SC functions at the NS7M station, reference waveforms to compute the dv/v,... (e.g., all the data required to reproduce the Figure).
  - The **ConvDeNoise_core.py** file is the convolutional denoising autoencoder main code that was used to compute the **ConvDeNoise_NS7M_station.h5** file (requires the raw SC functions, please email me for the training set, file is too big for Github)
 

# ECG Conformer snippet
ECG-based Hearth Arryhthmia Detection via Conformer Deep Neural Network
------------------------------------------------------------------------
The aim of this work was to train the conformer network with the dataset from the cooperation of the Chapman University USA with the Shaoxing People’s Hospital and consequently evaluate the classification performance with respect to seven heart rhythms. This classification, which stems from the examination of ECG measurements by medical professionals, is resource intensive. Should such a network detect heart rhythms with a high accuracy, it could be applied to relieve the burden on medical professionals. In this work the ECG dataset from Zheng et al., which contains 12 channel ECG measurements of 10646 patients and assignations to 11 heart rhythms by licensed physicians, was used. Wit this quantity and a sampling rate of 500 Hz, it quantitatively and qualitatively surpasses many other ECG datasets referenced in this work. The conformer was previously applied in the field of automatic speech recognition and accordingly had to be adapted for application on ECG data. Using this network, experiments with preprocessing, loss functions and adjustments to the dataset were performed to investigate classification performance. It was shown that the introduction of a uniform distribution, generated by oversampling, and the application of the MaxPooling layer instead of the Convolutional Subsampling layer, resulted in an improvement of the categorical accuracy of the network to 90.0%. In addition, it was shown that compositing the classes AF and AFIB, as well as the classes SA and SR, resulted in a further improvement of the categorical accuracy to 91.6%

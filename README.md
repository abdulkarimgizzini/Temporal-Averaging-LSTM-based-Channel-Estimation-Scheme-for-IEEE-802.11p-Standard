
This repository includes the source code of the LSTM-based channel estimators proposed in "Temporal Averaging LSTM-based Channel Estimation Scheme for IEEE 802.11 p Standard" paper that is published in the proceedings of the IEEE GLOBECOM 2022 conference that was held in Madrid (Spain). Please note that the Tx-Rx OFDM processing is implemented in Matlab (Matlab_Codes) and the LSTM processing is implemented in python (PyTorch) (Python_Codes).


### Matlab_Codes
1. Main_Simulation_Training.m: Run the Tx-Rx OFDM using a specific vehicular channel model, training SNR, and modulation order. In this file DPA and DPA_TA conventional channel estimators are implemented and saved to be used in the LSTM processing later (see the paper for detalied information). In order to save processing time, this file will run the simulation for Training_Size iterations only (Forexample if we are using 10000 channel realizations, instead of runnig the simulations for 10000 channel realization for each SNR, we just use here Training_Size = 8000). Moreover, we just save H_DPA, H_DPA_TA and the true channels. (No need to save the received frames and the Tx-bits here).

2. Main_Simulation_Testing.m: Run the Tx-Rx OFDM similarly as performed in (1) but here for Testing_Size iterations and for the whole SNR region and not for the training SNR only as considered in (1). You will notice here that we save all the simulation parameters including the received frames and the Tx-bits, in order to use them later in the LSTM results processing.

3. DG_LSTM_MLP_Training.m: This script will generte the training dataset obtained by (1). 

4. DG_LSTM_MLP_Testing.m: This script will generte the testing dataset obtained by (2).

5. LSTM_MLP_RP.m: After exeuting the LSTM processing using PyTorch, this script is used to process the LSTM results and calculate the BER and NMSE for each LSTM-based estimator.

6. DPA.m: Implement the conventional DPA estimation.

7. DPA_TA.m: Impelement the convcentional DPA + Time averaging estimation.

8. Channel_functions.m: Define the several vehicular channel models (see the paper for detalied information).

9. IDX_Generation.m: Generate the training and testing indices vectors. Here you just need to specify the data set size, training and testing datsets percentages.

### Python_Codes 

1. LSTM_DNN_DPA_Training.py: Train the LSTM-DNN-DPA estimator using H_DPA as an input and the corresponding true channels as an output.

2. LSTM_DPA_TA_Training.py: Train the LSTM-DPA-TA estimator using H_DPA_TA as an input and the corresponding true channels as an output.

3. LSTM_DNN_DPA_Testing.py: Test the LSTM-DNN-DPA estimator using the trained LSTM-DNN model in (1).  

4. LSTM_DPA_TA_Testing.py: Test the LSTM-DPA-TA estimator using the trained LSTM model in (2).  

5. functions.py: Implement the modulation/demodulation operations in python. 

For more information and questions, please contact me on abdulkarim.gizzini@ensea.fr

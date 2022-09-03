clc;clearvars;close all; warning('off','all');
% Load pre-defined DNN Testing Indices
load('./samples_indices_100.mat');
configuration = 'testing'; % training or testing
% Define Simulation parameters
nSC_In                    = 104;% 104
nSC_Out                   = 96;
nSym                      = 20;
mobility                  = 'High';
modu                      = 'QPSK';
ChType                    = 'VTV_SDWW';
scheme                    = 'DPA_TA';

ppositions             = [7,21, 32,46].';  
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';

if (isequal(configuration,'training'))
    indices = training_samples;
    EbN0dB           = 40; 
   
elseif(isequal(configuration,'testing'))
    indices = testing_samples;
    EbN0dB           = 0:5:40;    
end
Dataset_size     = size(indices,1);



SNR                       = EbN0dB.';
N_SNR                     = length(SNR);
for n_snr = 1:N_SNR
load(['./',mobility,'_',ChType,'_',modu,'_',configuration,'_simulation_' num2str(EbN0dB(n_snr)),'.mat'],...
'True_Channels_Structure', [scheme '_Structure'],'HLS_Structure');
    
Dataset_X        = zeros(nSC_In,nSym,Dataset_size);
Dataset_Y        = zeros(nSC_Out,nSym,Dataset_size);

True_Channels_Structure = True_Channels_Structure(:,2:end,:);    
scheme_Channels_Structure = eval([scheme '_Structure']);

RHP = real(scheme_Channels_Structure(:,1:end-1,:));
IHP = imag(scheme_Channels_Structure(:,1:end-1,:));

RPP = real(scheme_Channels_Structure(ppositions,1:end-1,:));
IPP = imag(scheme_Channels_Structure(ppositions,1:end-1,:));

if (isequal(configuration,'training'))
    
    Dataset_X(:,1,:)     = [real(HLS_Structure); imag(HLS_Structure)];
    Dataset_X(:,2:end,:) = [RHP; IHP];


    Dataset_Y(1:48,:,:)  = real(True_Channels_Structure(dpositions,:,:));
    Dataset_Y(49:96,:,:) = imag(True_Channels_Structure(dpositions,:,:));

    Dataset_X = permute(Dataset_X, [3, 2 ,1 ]);
    Dataset_Y = permute(Dataset_Y, [3 2 1]);

    LSTM_Datasets.('Train_X') =  Dataset_X;
    LSTM_Datasets.('Train_Y') =  Dataset_Y;
elseif(isequal(configuration,'testing'))
    
    load(['./',mobility,'_',ChType,'_',modu,'_',configuration,'_simulation_' num2str(EbN0dB(n_snr)),'.mat'],...
    'Received_Symbols_FFT_Structure');

    Received_Symbols_FFT_Structure  =  Received_Symbols_FFT_Structure(dpositions,:,:);
    Dataset_X(:,1,:) = [real(HLS_Structure); imag(HLS_Structure)];
    Dataset_X(ppositions,2:end,:) = RPP;
    Dataset_X(ppositions + 52,2:end,:) = IPP;
    
    Dataset_Y(1:48,:,:)  = real(True_Channels_Structure(dpositions,:,:));
    Dataset_Y(49:96,:,:) = imag(True_Channels_Structure(dpositions,:,:));

    Dataset_X = permute(Dataset_X, [3, 2 ,1 ]);
    Dataset_Y = permute(Dataset_Y, [3 2 1]);
    Received_Symbols_FFT_Structure = permute(Received_Symbols_FFT_Structure,[3 2 1]);
    LSTM_Datasets.('Test_X') =  Dataset_X;
    LSTM_Datasets.('Test_Y') =  Dataset_Y;  
    LSTM_Datasets.('Y_DataSubCarriers') =  Received_Symbols_FFT_Structure;
end

save(['./',mobility,'_',ChType,'_',modu,'_',scheme,'_LSTM_',configuration,'_dataset_' num2str(EbN0dB(n_snr)),'.mat'],  'LSTM_Datasets');

end
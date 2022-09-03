import time
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy
import pickle
from scipy.io import loadmat
import functions as fn
import os

class LSTM(nn.Module):
    def __init__(self, input_size, lstm_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstmcell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.lstm_size)
        self.out = nn.Sequential(nn.Linear(128, 96))

    def forward(self, x_cur, h_cur=None, c_cur=None):
        batch_size, _ = x_cur.size()
        if h_cur is None and c_cur is None:
            h_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
            c_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
        h_next, c_next = self.lstmcell(x_cur, (h_cur, c_cur))
        out = self.out(h_next)
        return out, h_next, c_next


def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error


def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse


# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)
# The following ratios is for splitting the training dataset into training and validation subsets
train_rate = 0.75
val_rate = 0.25

dposition = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,88,89,90,91,92,93,94,95,96,97,98,99,100,102,103,104,105,106,107]
dposition_WCP = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,59,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,91,92,93,94,95,96,98,99,100,101,102,103]
dposition_WP = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                 30,31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,83, 84, 85, 86,
                 87, 88, 89, 90, 91, 92, 93, 94, 95]

DSC_IDX = np.array(dposition_WCP)
ppos = [6,20,31,45,58,72,83,97]
PSC_IDX_Expanded = np.array(ppos)

if configuration_mode == 10:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    input_size = sys.argv[6]
    lstm_size = sys.argv[7]
    EPOCH = sys.argv[8]
    BATCH_SIZE = sys.argv[9]

    mat = loadmat('./{}_{}_{}_{}_LSTM_training_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, training_snr))
    Dataset = mat['LSTM_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)  # Size: Training_Samples x 2Kon
    print('Loaded Dataset Outputs: ', Y.shape)  # Size: Training_Samples x 2Kon

    # Reshape Input and Label Data
    input_data_Re = X.reshape(-1, 2)
    label_data_Re = Y.reshape(-1, 2)
    print('Reshaped Training Input Dataset: ', input_data_Re.shape)
    print('Reshaped Training Label Dataset: ', label_data_Re.shape)

    # Normalization
    scaler = StandardScaler()
    input_data_sclar = scaler.fit_transform(input_data_Re)  # .reshape(input_data.shape)
    label_data_sclar = scaler.fit_transform(label_data_Re)  # .reshape(label_data.shape)

    # Reshape after normalization
    input_data_sclar = input_data_sclar.reshape(X.shape)
    label_data_sclar = label_data_sclar.reshape(Y.shape)
    print('Reshaped Normalized Training Input Dataset: ', input_data_sclar.shape)
    print('Reshaped Normalized Training Label Dataset: ', label_data_sclar.shape)

    # Training and Validation Datasets splits
    nums = X.shape[0]
    train_nums = int(train_rate * nums)
    val_nums = int(nums * val_rate)
    print('dataset size: ', nums, ', train set size: ', train_nums, ', val set size: ', val_nums)

    # Assign training data set and validation data set
    Train_X = input_data_sclar[:train_nums]
    Train_Y = label_data_sclar[:train_nums]
    Val_X = input_data_sclar[-val_nums:]
    Val_Y = label_data_sclar[-val_nums:]
    print('Train_X :', Train_X.shape)
    print('Train_Y :', Train_Y.shape)
    print('Val_X :', Val_X.shape)
    print('Val_Y :', Val_Y.shape)
    train_input = torch.from_numpy(Train_X).type(torch.FloatTensor)
    train_label = torch.from_numpy(Train_Y).type(torch.FloatTensor)
    val_input = torch.from_numpy(input_data_sclar[-val_nums:]).type(torch.FloatTensor)
    val_label = torch.from_numpy(label_data_sclar[-val_nums:]).type(torch.FloatTensor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ---------------- generate batch dataset ------------------- #
    dataset = data.TensorDataset(train_input, train_label)
    loader = data.DataLoader(dataset=dataset, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=8 if torch.cuda.is_available() else 0)

    # ---------------------- train the model ------------------------ #
    r_min_err = float('inf')
    model = LSTM(int(input_size), int(lstm_size)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss()

    LOSS_TRAIN = []
    LOSS_VAL = []
    STEP = 0

    min_err = float('inf')
    time_train = 0

    for epoch in range(int(EPOCH)):
        # ---------------------- train ------------------------ #
        start = time.time()
        with torch.set_grad_enabled(True):
            scheduler.step()
            model.train()
            for step, (train_batch, label_batch) in enumerate(loader):
                train_batch, label_batch = train_batch.to(device), label_batch.to(device)
                optimizer.zero_grad()

                output = torch.zeros_like(label_batch)
                for t in range(train_batch.size(1)):
                    if t == 0:
                        out_t, hn, cn = model(train_batch[:, t, :])
                    else:
                        out_t, hn, cn = model(train_batch[:, t, :], hn, cn)
                    output[:, t, :] = out_t
                loss = criterion(output, label_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1e-4)
                optimizer.step()

                avg_err, s_err, error = calc_error(output.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
                if step % 200 == 0:
                    print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.item(), '| err: ', avg_err)
                    LOSS_TRAIN.append(loss)

        time_train += time.time() - start

        # ---------------------- validation ------------------------ #
        with torch.set_grad_enabled(False):
            model.eval()
            val_input, val_label = val_input.to(device), val_label.to(device)
            output = torch.zeros_like(val_label)
            for t in range(val_input.size(1)):
                if t == 0:
                    val_t, hn, cn = model(val_input[:, t, :])
                else:
                    val_t, hn, cn = model(val_input[:, t, :], hn, cn)
                output[:, t, :] = val_t

            loss = criterion(output, val_label)

            avg_err, s_err, error = calc_error(output.detach().cpu().numpy(), val_label.detach().cpu().numpy())
            print('Epoch: ', epoch, '| val err: ', avg_err)
            LOSS_VAL.append(loss)

            out1 = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
            val_label1 = scaler.inverse_transform(val_label.detach().cpu().numpy().reshape(-1, 2)).reshape(val_label.shape)

            if avg_err < min_err:
                min_err = avg_err
                best_model_wts = copy.deepcopy(model.state_dict())

    if min_err < r_min_err:
        r_min_err = min_err
        r_best_model_wts = best_model_wts

    model.load_state_dict(r_best_model_wts)
    torch.save(model.to('cpu'), './{}_{}_{}_{}_LSTM_{}.pkl'.format(mobility, channel_model, modulation_order, scheme, training_snr))
    plt.figure(1)
    x = range(int(EPOCH))
    plt.semilogy(x, LOSS_TRAIN, 'r-', label='loss_train')
    plt.semilogy(x, LOSS_VAL, 'b-', label='loss_val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

else:
    # We are running the testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]
    if modulation_order == 'QPSK':
        modu_way = 1
    elif modulation_order == '16QAM':
        modu_way = 2

    for n_snr in SNR_index:
        mat = loadmat('./{}_{}_{}_{}_LSTM_testing_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, n_snr))
        Dataset = mat['LSTM_Datasets']
        Dataset = Dataset[0, 0]
        X = Dataset['Test_X']
        Y = Dataset['Test_Y']
        yf_d = Dataset['Y_DataSubCarriers']
        print('Loaded Dataset Inputs: ', X.shape)
        print('Loaded Dataset Outputs: ', Y.shape)
        print('Loaded Testing OFDM Frames: ', yf_d.shape)
        hf_DL = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
        device = torch.device("cpu")
        NET = torch.load('./{}_{}_{}_{}_LSTM_{}.pkl'.format(mobility, channel_model, modulation_order, scheme, testing_snr)).to(device)
        scaler = StandardScaler()

        # For over all Frames
        for i in range(yf_d.shape[0]):
            hf = X[i, 0, :]
            hn, cn = None, None
            print('Testing Frame | ', i)
            # For over OFDM Symbols
            for j in range(yf_d.shape[1]):
                hf_input = hf
                input1 = scaler.fit_transform(hf_input.reshape(-1, 2)).reshape(hf_input.shape)
                input2 = torch.from_numpy(input1).type(torch.FloatTensor).unsqueeze(0)
                output, hn, cn = NET(input2.to(device), hn, cn)  # ([1,96])
                out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
                hf_out = out[:, :48] + 1j * out[:, 48:]  # (1,48)
                hf_DL[i, j, :] = hf_out
                sf = yf_d[i, j, :] / hf_out  # (1,48)
                x = fn.demap(sf, modu_way)
                xf = fn.map(x, modu_way)
                hf_out = yf_d[i, j, :] / xf
                hf_out = hf_out.ravel()
                if j < yf_d.shape[1] - 1:
                    hf_out_Expanded = np.concatenate((hf_out.real, hf_out.imag), axis=0)
                    X[i, j + 1, DSC_IDX] = hf_out_Expanded
                    hf = 0.5 * hf + 0.5 * X[i, j + 1, :]
        # Save Results
        result_path = './{}_{}_{}_{}_LSTM_Results_{}.pickle'.format(mobility, channel_model, modulation_order, scheme, n_snr)
        dest_name = './{}_{}_{}_{}_LSTM_Results_{}.mat'.format(mobility, channel_model, modulation_order, scheme, n_snr)
        with open(result_path, 'wb') as f:
            pickle.dump([X, Y, hf_DL], f)

        a = pickle.load(open(result_path, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_test_x_{}'.format(scheme, n_snr): a[0],
            '{}_test_y_{}'.format(scheme, n_snr): a[1],
            '{}_corrected_y_{}'.format(scheme, n_snr): a[2]
        })
        print("Data successfully converted to .mat file ")
        os.remove(result_path)





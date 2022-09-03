import torch
from sklearn.preprocessing import StandardScaler
import math
import numpy as np


def map(signal_bit, modu_way): 
    '''
    :param signal_bit: the bit signal ,shape = (ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output , pilot_symbol
             output = signal_symbol, shape =(ofdm_sym_num, data_sub_num)
    '''

    if modu_way == 0:
        output = map_bpsk(signal_bit)
    elif modu_way == 1:
        output = map_qpsk(signal_bit)
    elif modu_way == 2:
        output = map_16qam(signal_bit)
    elif modu_way == 3:
        output = map_64qam(signal_bit)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def map_bpsk(signal_bit):
    output = np.empty_like(signal_bit, dtype="complex64")
    for m in range(signal_bit.shape[0]):
        for n in range(signal_bit.shape[1]):
            if signal_bit[m, n] == 0:
                output[m, n] = -1 + 0j
            else:
                output[m, n] = 1 + 0j
    return output


def map_qpsk(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1] / 2)
    x = signal_bit.reshape(c, d, 2)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :]
            if (a == [0, 0]).all():
                output[m, n] = complex(-math.sqrt(2)/2, -math.sqrt(2)/2)
            elif (a == [0, 1]).all():
                output[m, n] = complex(-math.sqrt(2)/2, math.sqrt(2)/2)
            elif (a == [1, 1]).all():
                output[m, n] = complex(math.sqrt(2) / 2, math.sqrt(2) / 2)
            else:
                output[m, n] = complex(math.sqrt(2) / 2, -math.sqrt(2) / 2)
    return output


def map_16qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/4)
    x = signal_bit.reshape(c, d, 4)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :2]
            if (a == [0, 0]).all():
                real = -3
            elif (a == [0, 1]).all():
                real = -1
            elif (a == [1, 1]).all():
                real = 1
            else:
                real = 3
            b = x[m, n, 2:]
            if (b == [0, 0]).all():
                imag = -3
            elif (b == [0, 1]).all():
                imag = -1
            elif (b == [1, 1]).all():
                imag = 1
            else:
                imag = 3
            output[m, n] = complex(real, imag)/math.sqrt(10)
    return output


def map_64qam(signal_bit):
    c = int(signal_bit.shape[0])
    d = int(signal_bit.shape[1]/6)
    x = signal_bit.reshape(c, d, 6)
    output = np.empty((c, d), dtype="complex64")
    for m in range(c):
        for n in range(d):
            a = x[m, n, :3]
            if (a == [0, 0, 0]).all():
                real = -7
            elif (a == [0, 0, 1]).all():
                real = -5
            elif (a == [0, 1, 1]).all():
                real = -3
            elif (a == [0, 1, 0]).all():
                real = -1
            elif (a == [1, 0, 0]).all():
                real = 7
            elif (a == [1, 0, 1]).all():
                real = 5
            elif (a == [1, 1, 1]).all():
                real = 3
            else:
                real = 1
            b = x[m, n, 3:]
            if (b == [0, 0, 0]).all():
                imag = -7
            elif (b == [0, 0, 1]).all():
                imag = -5
            elif (b == [0, 1, 1]).all():
                imag = -3
            elif (b == [0, 1, 0]).all():
                imag = -1
            elif (b == [1, 0, 0]).all():
                imag = 7
            elif (b == [1, 0, 1]).all():
                imag = 5
            elif (b == [1, 1, 1]).all():
                imag = 3
            else:
                imag = 1
            output[m, n] = complex(real, imag)/math.sqrt(84)
    return output


def demap(signal_symbol, modu_way):
    '''
    :param signal_symbol: the symbol signal ,shape = (ofdm_sym_num, data_sub_num)
    :param modu_way:  0:bpsk, 1:qpsk, 2:16qam, 3:64qam
    :return: output
             output = signal_bit, shape =(ofdm_sym_num, data_sub_num*bit_to_sym[modu_way])
    '''
    if signal_symbol.ndim == 1:
        signal_symbol = signal_symbol[np.newaxis, :]
    if modu_way == 0:
        output = demap_bpsk(signal_symbol)
    elif modu_way == 1:
        output = demap_qpsk(signal_symbol)
    elif modu_way == 2:
        output = demap_16qam(signal_symbol)
    elif modu_way == 3:
        output = demap_64qam(signal_symbol)
    else:
        print('the input of modu_way is error')
        output = 1
    return output


def demap_bpsk(x):
    output = np.empty_like(x, dtype="int")
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if x[m, n].real >= 0:
                output[m, n] = 1
            else:
                output[m, n] = 0
    return output


def demap_qpsk(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 2), dtype="int")
    for m in range(c):
        for n in range(d):
            a = x[m, n].real
            b = x[m, n].imag
            if (a <= 0) & (b <= 0):
                output[m, n, :] = [0, 0]
            elif (a <= 0) & (b > 0):
                output[m, n, :] = [0, 1]
            elif (a > 0) & (b > 0):
                output[m, n, :] = [1, 1]
            else:
                output[m, n, :] = [1, 0]
    output = output.reshape(c, int(2*d))
    return output


def demap_16qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 4), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(10)*x[m, n].real
            if a <= -2:
                output[m, n, :2] = [0, 0]
            elif (a <= 0) & (a > -2):
                output[m, n, :2] = [0, 1]
            elif (a <= 2) & (a > 0):
                output[m, n, :2] = [1, 1]
            else:
                output[m, n, :2] = [1, 0]
            b = math.sqrt(10)*x[m, n].imag
            if b <= -2:
                output[m, n, 2:] = [0, 0]
            elif (b <= 0) & (b > -2):
                output[m, n, 2:] = [0, 1]
            elif (b <= 2) & (b > 0):
                output[m, n, 2:] = [1, 1]
            else:
                output[m, n, 2:] = [1, 0]
    output = output.reshape((c, int(4*d)))
    return output


def demap_64qam(x):
    c = int(x.shape[0])
    d = int(x.shape[1])
    output = np.empty((c, d, 6), dtype="int")
    for m in range(c):
        for n in range(d):
            a = math.sqrt(84)*x[m, n].real
            if a <= -6:
                output[m, n, :3] = [0, 0, 0]
            elif (a > -6) & (a <= -4):
                output[m, n, :3] = [0, 0, 1]
            elif (a > -4) & (a <= -2):
                output[m, n, :3] = [0, 1, 1]
            elif (a > -2) & (a <= 0):
                output[m, n, :3] = [0, 1, 0]
            elif (a > 0) & (a <= 2):
                output[m, n, :3] = [1, 1, 0]
            elif (a > 2) & (a <= 4):
                output[m, n, :3] = [1, 1, 1]
            elif (a > 4) & (a <= 6):
                output[m, n, :3] = [1, 0, 1]
            else:
                output[m, n, :3] = [1, 0, 0]
            b = math.sqrt(84) * x[m, n].imag
            if b <= -6:
                output[m, n, 3:] = [0, 0, 0]
            elif (b > -6) & (b <= -4):
                output[m, n, 3:] = [0, 0, 1]
            elif (b > -4) & (b <= -2):
                output[m, n, 3:] = [0, 1, 1]
            elif (b > -2) & (b <= 0):
                output[m, n, 3:] = [0, 1, 0]
            elif (b > 0) & (b <= 2):
                output[m, n, 3:] = [1, 1, 0]
            elif (b > 2) & (b <= 4):
                output[m, n, 3:] = [1, 1, 1]
            elif (b > 4) & (b <= 6):
                output[m, n, 3:] = [1, 0, 1]
            else:
                output[m, n, 3:] = [1, 0, 0]
    output = output.reshape(c, int(6*d))
    return output





def sta_dnn(hf_p_ls, yf_d, d_index, modu_way, v_index, c_index):
    frame_num = yf_d.shape[0]
    frame_len = yf_d.shape[1]
    hf_sta_dnn = np.zeros((frame_num, frame_len, 64), dtype="complex64")
    sf = np.zeros((1, d_index.size), dtype="complex64")
    hf = np.zeros(yf_d.shape[2], dtype="complex64")
    hf_update = np.zeros(64, dtype="complex64")
    device = torch.device("cpu")
    NET4 = 1
    scaler = StandardScaler()
    for i in range(frame_num):
        hf[d_index] = np.mean(hf_p_ls[i, :2, d_index], axis=1)
        for j in range(frame_len):
            hf_sta_dnn[i, j, :] = hf
            sf[:, :] = yf_d[i, j, d_index] / hf[d_index]
            x = demap(sf, modu_way)
            xf = map(x, modu_way)
            hf[d_index] = yf_d[i, j, d_index] / xf
            hf[c_index] = hf_p_ls[i, j + 2, c_index]
            for k in range(yf_d.shape[2]):
                a, b = 0, 0
                for m in range(-2, 3, 1):  # 由于边界和空符号的影响需要分段考虑
                    if k + m >= 6 and k + m < 59 and k + m != 32:
                        a = a + hf[k + m]
                        b = b + 1  # 计数符号，计加了几次
                if b != 0:
                    hf_update[k] = a / b
            hf = 1 / 2 * hf_update + 1 / 2 * hf
            input = np.concatenate((hf[v_index].real, hf[v_index].imag), axis=0)
            # ----------------实例化------------------#
            input1 = scaler.fit_transform(input.reshape(-1, 2)).reshape(input.shape)
            input2 = torch.from_numpy(input1).type(torch.FloatTensor)
            _, output = NET4(input2.to(device))
            out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
            hf[d_index] = out[:48] + 1j * out[48:]
    return hf_sta_dnn





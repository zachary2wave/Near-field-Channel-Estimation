import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from basic_parameter import *
from torch.utils.data import DataLoader
from util.util_function import CustomDataset
import scipy.io as io


def DMA_Steering_Vector(sin, d):

    '''
    Function:
        Generate
    Parameters:
        param nx, ny: number of antennas in x direction and y direction
        param az and el: vector of sin(theta) where theta is the azimuth angle,
                         vector of sin(theta) where theta is the Elevation angle
        param d: distance between two adjacent antennas
        param lamb: signal wave length
    Return:
         Matrix with each column an array response vector
    '''

    vi = np.array([-1j], dtype=complex)
    cos = np.sqrt(1-sin**2)
    dis = np.zeros(N)
    ARV = np.zeros(N, dtype=complex)
    for i in range(N):
        dis[i] = np.sqrt((d * sin - array_position[i, 0]) ** 2 + (d * cos - array_position[i, 1]) ** 2)
        ARV[i] = np.exp(vi * k * dis[i])[0]
    return ARV

def directionary():
    dictionary = np.zeros((N, Num_grid_angle * Num_grid_d), dtype=np.complex64)
    sensing_matrix = np.zeros((N, Num_grid_angle * Num_grid_d), dtype=np.complex64)
    record_area = np.zeros((Num_grid_angle * Num_grid_d, 2))
    index = 0
    for i in range(Num_grid_angle):
        for j in range(Num_grid_d):
            ARV = DMA_Steering_Vector(grid_angle[i], grid_d[j])
            dictionary[:, index] = ARV
            record_area[index, 0] = grid_d[j]
            record_area[index, 1] = grid_angle[i]
            index += 1
    sensing_matrix = np.matmul(h_random, dictionary)
    return dictionary, sensing_matrix, record_area

# G = DMA_H_matrix(Nd, Ne)
dic, sensing_matrix, record_area = directionary()
# dic, corresponding_index, basis_matrix = directionary_orthogonality()
dic_H = np.conj(dic.T)
real = torch.from_numpy(dic.real).float()
imag = torch.from_numpy(dic.imag).float()
dic_tensor = torch.complex(real, imag)
dic_tensor = dic_tensor.cuda()


dpinv = np.linalg.pinv(dic)



real = torch.from_numpy(sensing_matrix.real).float()
imag = torch.from_numpy(sensing_matrix.imag).float()
sensing_matrix_tensor = torch.complex(real, imag)
sensing_matrix_tensor = sensing_matrix_tensor.cuda()


Num_grid = dic.shape[1]


def Data_generate_on_grid(SNR, h_random, on_grid=True):
    '''
    :return:
        channel,  dim =  [Ne, Nd]
        information, all middle variable
    '''

    SNR = 10 ** (SNR / 10)
    path = np.random.randint(low=2, high=6)
    H = 0
    HH1, HH2 = 0, 0
    dis = np.zeros((path))
    grid_num = np.zeros((path),dtype=np.int32)
    angle = np.zeros((path, 2))
    F = np.zeros((path))
    Am, Ph = np.zeros((path)), np.zeros((path, N), dtype=np.complex64)
    S = np.zeros([Num_grid])
    for i in range(0, path):
        if on_grid:
            grid_num[i] = int(np.random.randint(0, Num_grid))
            sin = grid_angle[grid_num[i] // Num_grid_d]
            d = grid_d[grid_num[i] % Num_grid_d]
            F[i] = 6 * (sin ** 2)
            Am[i] = np.sqrt(F[i]) * La / (4 * np.pi * d)
            H += Am[i] * dic[:, int(grid_num[i])]
            S[int(grid_num[i])] = Am[i]
        else:
            am = np.random.uniform()
            if am < 0.2:
                sin = np.clip(np.random.normal(0.3, 0.15), -1, 1)
            elif 0.2 < am < 0.4:
                sin = np.clip(np.random.normal(0.6, 0.15), -1, 1)
            elif 0.4 < am < 0.6:
                sin = np.clip(np.random.normal(-0.2, 0.15), -1, 1)
            elif 0.6 < am < 0.8:
                sin = np.clip(np.random.normal(-0.45, 0.15), -1, 1)
            else:
                sin = np.clip(np.random.normal(-0.6, 0.15), -1, 1)


            d = np.random.uniform(low=0.01, high=0.45, size=[1])
            d = 1/d

            Am[i] = La / (4 * np.pi * d)
            Ph = DMA_Steering_Vector(sin, d)
            H += Am[i] * Ph
            # now_gird = np.array([d, sin])
            # index_AZ = np.argmin(np.sum(np.abs(record_area - now_gird), 1))
            # Phnew = DMA_Steering_Vector(record_area[index_AZ, 1], record_area[index_AZ, 0])
            # PPPH = np.abs(np.matmul(np.conj(Ph[np.newaxis,:]), dic))
            # indexzz = np.argmax(PPPH)
            # maxcor = np.max(PPPH)
            #
            # HH1 += Am[i] * dic[:, int(indexzz)]
            # HH2 += Am[i] * dic[:, int(index_AZ)]
            # S[int(indexzz)] = Am[i]
    # HH = np.matmul(dpinv, H[:, np.newaxis])
    # HH = np.matmul(dic, HH)/N
    # error1 = np.sum((H.real - HH1.real) ** 2 + (H.imag - HH1.imag) ** 2) / np.sum(H.real ** 2 + H.imag ** 2)
    # error2 = np.sum((H.real - HH2.real) ** 2 + (H.imag - HH2.imag) ** 2) / np.sum(H.real ** 2 + H.imag ** 2)
    S = np.matmul(dpinv, H)
    S_abs = np.abs(S)
    S_sort = np.sort(S_abs)[-2*path]
    zeros_place = S_abs<S_sort
    S[zeros_place] = 0
    # HH2 = np.matmul(dic, S)
    # error2 = np.sum((H.real - HH2.real) ** 2 + (H.imag - HH2.imag) ** 2) / np.sum(H.real ** 2 + H.imag ** 2)

    signal = np.sqrt(np.conj(H.T).dot(H))
    real_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    imag_part = np.expand_dims(1/np.sqrt(2)*np.random.normal(0, 1, size=(N, )), axis=-1)
    noise = np.concatenate((real_part, imag_part), axis=-1).view(np.complex).squeeze(-1)/np.sqrt(N)
    alpha = np.sqrt(SNR)
    Y_ = alpha * H / (1+alpha) + noise * signal / (1+alpha)
    Y = h_random.dot(Y_)
    Y = Y * 1e3
    S = S * 1e3
    return Y, S, H, Y_


def generate_sample(data_size, snr, random_weight = h_random, gird = True):

    Y = np.zeros([data_size, N_RF], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):

        Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_on_grid(snr, random_weight, gird)
        # if i%data_size==0:
        #     print("now generate the", i // data_size, "th batch data")
    data_set = CustomDataset(label, Y, H, Y_)
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)



def generate_dataset_random_snr(data_size, random_weight = h_random, gird = True):
    Y = np.zeros([data_size, N_RF], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_ [i, :] = Data_generate_on_grid(snr, random_weight, gird)
    data_set = CustomDataset(label, Y, H, Y_ )
    return DataLoader(data_set, batch_size=batchsize, shuffle=True)

def save_dataset_random_snr(data_size, name, random_weight = h_random, gird = True ):
    Y = np.zeros([data_size, N_RF], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_on_grid(snr, random_weight, gird)
        if i % batchsize == 0:
            print("now generate the", i // batchsize, "th batch data")
    np.savez(name, Y=Y, label=label, H=H, noise_power=Y_, weight = random_weight)


def save_dataset_random_snr(data_size, name, random_weight = h_random, gird = True ):
    Y = np.zeros([data_size, N_RF], dtype=np.complex64)
    label = np.zeros([data_size, Num_grid], dtype=np.complex64)
    H = np.zeros([data_size, N], dtype=np.complex64)
    Y_ = np.zeros([data_size, N], dtype=np.complex64)
    for i in range(data_size):
        snr = np.random.randint(0, 30)
        Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_on_grid(snr, random_weight, gird)
        if i % batchsize == 0:
            print("now generate the", i // batchsize, "th batch data")
    np.savez(name, Y=Y, label=label, H=H, noise_power=Y_, weight = random_weight)

def save_dataset_matlab(data_size, random_weight, gird = True ):
    for snr in range(30):
        Y = np.zeros([data_size, N_RF], dtype=np.complex64)
        label = np.zeros([data_size, Num_grid], dtype=np.complex64)
        H = np.zeros([data_size, N], dtype=np.complex64)
        Y_ = np.zeros([data_size, N], dtype=np.complex64)
        for i in range(data_size):
            Y[i, :], label[i, :], H[i, :], Y_[i, :] = Data_generate_on_grid(snr, random_weight, gird)
            if i % batchsize == 0:
                print("now generate the", i // batchsize, "th batch data")
        io.savemat("Ongrid_snr%d.mat"%snr, {"dic": dic, "Q": random_weight,
                                     "Y": Y, "label": label, "Y_": Y_,
                                     "label_h": H})


if __name__=="__main__":

    # save_dataset_matlab(15 * batchsize, random_weight=h_random, gird=True)

    # save_dataset_matlab(15 * batchsize, random_weight=h_random, gird=True)
    # save_dataset_random_snr(1000 * batchsize, 'Sample_ongrid_6', random_weight = h_random, gird = True )
    # save_dataset_random_snr(1000 * batchsize, 'Sample_offgrid_new2', random_weight = h_random, gird = False)
    for _ in range(100):
        test_data1111 = generate_sample(batchsize, snr, random_weight=h_random, gird=False)


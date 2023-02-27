import sys
sys.path.append("/home/zhangxy/nearfield/")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from basic_parameter import *
from data_generate import *
from util.util_function import XXXtest_model_grad as Gcheck
from util.util_function import XXXtest_model as Mpara
torch.cuda.set_device(0)
device = torch.device("cuda:0")


'''
The file is for test the performance on 
+ non orthogonality Sparse representation basis
+ Without Noise
'''

from util import logger
import time

SNR = "random"
initial_etas = 1e-4
path = "/home/zhangxy/nearfield/Log_data_save/off_grid_LISTAv2/"
configlist = ["stdout", "csv", 'tensorboard']
logger.configure(path, configlist)

#
# path1 = os.path.abspath('..')
# train_data_set = np.load(path1+"\\Sample_offgrid.npz")
train_data_set = np.load("/home/zhangxy/nearfield/Sample_offgrid_6.npz")
data_set = CustomDataset(train_data_set["label"], train_data_set["Y"], train_data_set["H"], train_data_set["noise_power"])
train_data = DataLoader(data_set, batch_size=batchsize, shuffle=True)
DMA_random = train_data_set["weight"]
# train_data = generate_dataset_random_snr(500*batchsize, random_weight=DMA_random, gird=True)
test_data = generate_dataset_random_snr(batchsize, random_weight=DMA_random, gird=False)

## get the max eig
L = np.matmul(dic.transpose(), dic)
Lmax, _ = np.linalg.eig(L)

DMA_random_tensor = torch.from_numpy(h_random)
DMA_random_tensor = complex_feature_trasnsmission(DMA_random_tensor)
DMA_random_tensor = DMA_random_tensor.cuda()
from LISTA import LISTA_c

model = LISTA_c(Num_grid, N_RF, N)
model = model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=True
                                                       , threshold=10, threshold_mode='rel', cooldown=50,
                                                       min_lr=1e-6, eps=1e-08)
bestL = 1e5
steps = 0
for looptime in range(itertimes):

    for inloop, Dtrain in enumerate(train_data):
        Sendin, label, label_H = Dtrain[0], Dtrain[1], Dtrain[2]
        Sendin, label, label_H = Sendin.cuda(), label.cuda(), label_H.cuda()
        label = label.real

        S = model(Sendin)
        L = F.mse_loss(S[0], label) + F.l1_loss(S[0], label)
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, norm_type=2.0)
        optimizer.step()
        with torch.no_grad():
            truth_mes = F.mse_loss(S[0].mm(dic_tensor.real.T) / 1e6, label_H.real) + \
                        F.mse_loss(S[0].mm(dic_tensor.imag.T) / 1e6, label_H.imag)
            truth_mes = truth_mes / torch.mean(torch.sum(label_H.real ** 2 + label_H.imag ** 2, dim=1))

    test_data = generate_dataset_random_snr(batchsize, random_weight=DMA_random, gird=True)
    Dtest = next(iter(test_data))
    with torch.no_grad():
        Sendin_t, label_t, label_H_t = Dtest[0], Dtest[1], Dtest[2]
        Sendin_t, label_t, label_H_t = Sendin_t.cuda(), label_t.cuda(), label_H_t.cuda()
        S = model(Sendin_t)
        testL = F.mse_loss(S[0], label_t.real)

        truth_mse_t = F.mse_loss(S[0].mm(dic_tensor.real.T) / 1e6, label_H_t.real) + \
                      F.mse_loss(S[0].mm(dic_tensor.imag.T) / 1e6, label_H_t.imag)
        truth_mse_t = truth_mse_t / torch.mean(torch.sum(label_H_t.real ** 2 + label_H_t.imag ** 2, dim=1))

        # scheduler.step(truth_mse_t)

        logger.record_tabular("steps", looptime)
        logger.record_tabular("Train_loss", float(L))
        logger.record_tabular("Truth_MSE", float(truth_mes)*128)

        logger.record_tabular("lr", optimizer.state_dict()["param_groups"][0]["lr"])
        logger.record_tabular("Test_loss", float(testL))
        logger.record_tabular("Test_Truth_MSE", float(truth_mse_t)*128)


        logger.dump_tabular()

        if looptime > 30 and float(truth_mse_t) < 0.9*bestL:
            bestL = float(truth_mse_t)
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, path + str(os.path.basename(__file__))[:-3] + ".pkl")
            print("the MSE have reached", float(truth_mse_t), "model has been saved")

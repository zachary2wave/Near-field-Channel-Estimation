import os
import sys

sys.path.append("/home/zhangxy/CE3/")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from data_generate import *
from util.util_function import sparse_degree


'''
The file is for test the performance on learning the sparse basis matrix 
'''

from util import logger
import time
K = 128
time_now = str(time.strftime("%m%d%H%M", time.localtime()))
path = "/home/zhangxy/nearfield/Log_data_save/off_grid_gridnum" + str(K) + "/"
configlist = ["stdout", "csv", 'tensorboard']
logger.configure(path, configlist)

# path1 = os.path.abspath('.')
# train_data_set = np.load(path1+"\\Sample_ongrid.npz")
train_data_set = np.load("/home/zhangxy/nearfield/Sample_offgrid.npz")
data_set = CustomDataset(train_data_set["label"], train_data_set["Y"], train_data_set["H"], train_data_set["noise_power"])
train_data = DataLoader(data_set, batch_size=batchsize, shuffle=True)
DMA_W = train_data_set["weight"]
test_data = generate_dataset_random_snr(batchsize, random_weight=h_random, gird=False)

DMA_random_tensor = torch.from_numpy(h_random)
DMA_random_tensor = complex_feature_trasnsmission(DMA_random_tensor)
DMA_random_tensor = DMA_random_tensor.cuda()
from ISTA_NET import ISTA_Net
model = ISTA_Net(K, N_RF, N, DMA_random_tensor)

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
        x, loss_sparse, loss_equal_map, out_sparse = model(Sendin)
        L = huber_loss(x[0], label_H.real*1e3) + huber_loss(x[1], label_H.imag*1e3)
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer.step()

        with torch.no_grad():
            truth_mse = F.mse_loss(x[0]/1e3, label_H.real)+\
                        F.mse_loss(x[1]/1e3, label_H.imag)
            truth_mse = truth_mse / torch.mean(torch.sum(label_H.real ** 2 + label_H.imag ** 2, dim=1))
            pass
        # logger.record_tabular("steps", steps)
        # logger.record_tabular("Loss", float(L))
        # logger.record_tabular("Loss_sparse_L1", float(loss_sparse))
        # logger.record_tabular("Loss_equal_map", float(loss_equal_map))
        # logger.record_tabular('sparse', out_sparse)
        # logger.record_tabular("MSE", float(truth_mse)*400)
        # logger.record_tabular("lr", optimizer.state_dict()["param_groups"][0]["lr"])
        # logger.dump_tabular()
        # pass

    Dtest = next(iter(test_data))
    with torch.no_grad():
        Sendin_t, label_t, label_H_t = Dtest[0], Dtest[1], Dtest[2]
        Sendin_t, label_t, label_H_t = Sendin_t.cuda(), label_t.cuda(), label_H_t.cuda()

        x, loss_sparse_t, loss_equal_map_t, out_sparse_t = model(Sendin_t)
        truth_mse_t = F.mse_loss(x[0]/ 1e3, label_H_t.real)+\
                F.mse_loss(x[1]/1e3, label_H_t.imag)
        truth_mse_t = truth_mse_t / torch.mean(torch.sum(label_H_t.real ** 2 + label_H_t.imag ** 2, dim=1))
        # scheduler.step(L)

        logger.record_tabular("steps", looptime)
        logger.record_tabular("Loss", float(L))
        logger.record_tabular("Loss_sparse_L1", float(loss_sparse))
        logger.record_tabular("Loss_equal_map", float(loss_equal_map))
        logger.record_tabular('sparse', out_sparse)

        logger.record_tabular("sparse_L1", float(loss_sparse_t))
        logger.record_tabular("equal", float(loss_equal_map_t))


        logger.record_tabular("sparse_drgeen", float(out_sparse_t))
        logger.record_tabular("sparse_drgeen_label", float(sparse_degree(label_t.real)))

        logger.record_tabular("MSE", float(truth_mse)*128)
        logger.record_tabular("MSE_test", float(truth_mse_t)*128)
        logger.record_tabular("lr", optimizer.state_dict()["param_groups"][0]["lr"])
        logger.dump_tabular()

        if looptime > 100 and float(truth_mse_t) < 0.9*bestL:
            bestL = float(truth_mse_t)
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, path + "model.pkl")
            print("the MSE have reached", float(truth_mse_t), "model has been saved")
        steps += 1
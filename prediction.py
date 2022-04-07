import pandas as pd
import numpy as np
import os

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
from utils.metrics import metric

args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

# TODO 换数据集需要修改名称
args.data = 'ETTh2ms1f2' # data
args.root_path = './data/ETT/'     # root path of data file
args.data_path = 'archxixia_ms.csv'        # data file
args.features = 'ms'           # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT'             # target feature in S or MS task
# U:microsecond
args.freq = 'ms'  # freq for time features encoding, options:[u: microsecondly, s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './informer_checkpoints'     # location of model checkpoints

# TODO NOTE important adding:
args.train_ratio = 0.8
args.dev_ratio = 0.1
args.test_ratio = 0.1


# TODO 修改三个参数大小
args.seq_len = 1152    # input sequence length of Informer encoder
args.label_len = 1152   # start token length of Informer decoder
args.pred_len = 576      # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 1 # encoder input size
args.dec_in = 1 # decoder input size
args.c_out = 1 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 3 # num of encoder layers
args.d_layers = 2 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob'  # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True    # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder
args.mix = True
args.padding = 0
args.freq = 'ms'

args.batch_size = 4
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False      # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 5
args.train_epochs = 6
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'xixia_ms.csv','T':'OT','M':[2,2,2],'S':[1,1,1],'MS':[2,2,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2ms1f2':{'data':'archxixia_ms.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1], 'ms':[1,1,1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
# args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

"""
加载训练好的模型预测，预测是一个与数据最后日期相邻的序列，在数据中不存在
如果想获取更多关于预测的信息，可以参考代码`exp/exp_informer.py function predict()`和`data/data_loader.py class Dataset_Pred`
"""
# 这里给出predict()函数详细代码
def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')
    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        # 模型在gpu训练，使用cpu加载
        exp.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        # exp.model.load_state_dict(torch.load(best_model_path))
    exp.model.eval()

    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        """
        batch_x:[1,1152,1]
        batch_x_mark:[1,1152,7]
        batch_y:[1,1152,1]
        batch_y_mark:[1,1728,7]    1728=1152+576(预测的长度)
        """
        # decoder input
        # dec_inp:[1,1728,1]   1152+576(后面全都填充0)
        if exp.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    # outputs[1,576,1]
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features =='MS' else 0
        batch_y = batch_y[:, -exp.args.pred_len:,f_dim].to(exp.device)

        pred = outputs.detach().cpu().numpy()
        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    # 预测的结果保存在 ./results/{setting}/real_prediction.npy
    folder_path = './results/'+setting+'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path+ 'real_prediction.npy', preds)

    return preds

# you can also use this prediction function to get result

# set saved model path
setting = 'informer_ETTh2ms1f2_ftms_sl1152_ll1152_pl576_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
exp = Exp(args)

prediction = predict(exp, setting, True)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(prediction[0,:,0])
# plt.savefig('pred.png')
plt.show()


# ******************************************************************************Test*******************************************************************************************
def test(exp, setting, load=False):
    test_data, test_loader = exp._get_data(flag='test')
    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        # exp.model.load_state_dict(torch.load(best_model_path))
    exp.model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        pred, true = exp._process_one_batch(
            test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return
# print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
# test(exp, setting, True)

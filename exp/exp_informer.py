from data.data_loader import Dataset_ETT_ms, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

from utils.r_drop import RDrop

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            #import ipdb; ipdb.set_trace()
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        # print(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'ETTh2ms1f2': Dataset_ETT_ms, # -> ms 
        }
        Data = data_dict[self.args.data] # data.data_loader.Dataset_ETT_ms
        timeenc = 0 if args.embed!='timeF' else 1 # timeenc=1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        #import ipdb; ipdb.set_trace()
        data_set = Data( # data.data_loader.Dataset_ETT_hour, the name of the class does not matter!
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag, # 'train'
            size=[args.seq_len, args.label_len, args.pred_len], # [336, 336, 168] -> [1152, 1152, 576]
            features=args.features, # "M" -> 'ms'
            target=args.target, # 'OT' = oil temperature
            inverse=args.inverse, # False
            timeenc=timeenc, # 1
            freq=freq, # 'h' -> 'ms'
            cols=args.cols, # None
            train_ratio = args.train_ratio,
            dev_ratio = args.dev_ratio,
            test_ratio = args.test_ratio,
        )
        #import ipdb; ipdb.set_trace()
        print(flag, len(data_set))   # "train 97805"; "val 11867"; "test 11867"
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # MAE损失函数
    def _select_criterion1(self):
        mae_criterion = nn.L1Loss()
        return mae_criterion

    # TODO 修改的新的损失函数
    def _select_criterion2(self):
        RDrop_criterion = RDrop()
        return RDrop_criterion


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        #import ipdb; ipdb.set_trace()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            pred1, true1 = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            # TODO 原文对回归任务，r_drop使用mse代替KL
            # Lmser = criterion(pred.detach().cpu(), pred1.detach().cpu())
            # Lmse = (criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(pred1.detach().cpu(), true.detach().cpu()))/2
            # loss = Lmse + Lmser
            # loss = criterion(pred.detach().cpu(), true.detach().cpu(), pred1.detach().cpu(), true1.detach().cpu())

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        #import ipdb; ipdb.set_trace()
        train_data, train_loader = self._get_data(flag = 'train')
        #import ipdb; ipdb.set_trace()
        vali_data, vali_loader = self._get_data(flag = 'val')
        #import ipdb; ipdb.set_trace()
        test_data, test_loader = self._get_data(flag = 'test')

        #import ipdb; ipdb.set_trace()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)   # 3056
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()
        # TODO 修改损失函数
        # criterion = self._select_criterion2()
        criterion = self._select_criterion1()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            #import ipdb; ipdb.set_trace()
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                #import ipdb; ipdb.set_trace()
                iter_count += 1
                
                model_optim.zero_grad()
                #import ipdb; ipdb.set_trace()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred1, true1 = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                # TODO 三种不同的损失函数计算
                # 原始的MSE损失函数
                loss = criterion(pred, true)

                # 原文对回归任务，r_drop使用mse代替KL
                # Lmser = criterion(pred, pred1)
                # Lmse = (criterion(pred, true) + criterion(pred1, true))/2
                # loss = Lmse + Lmser

                # r_drop使用KL计算
                # loss = criterion(pred, true, pred1, true1)
                # print(loss.shape)
                # print(loss.item())
                # print(type(loss))

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
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
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y

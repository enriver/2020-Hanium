# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:51:21 2020

@author: seungjun
"""

import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import os
from pathlib import Path 
import pickle
import random
from datetime import datetime, timedelta

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import ParameterGrid
from efficientnet_pytorch import EfficientNet
from fbprophet import Prophet

### torch
import torch.optim as optim
import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import shutil
from torch.utils.data import DataLoader

"""
mixed_precision=True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed
"""

##hand made
import pre_data
import print_Candle
import dataloader
from dataloader import Pathdataset, FBdataset
import models
from utils import Logger, AVERAGEMETER, savefig
import DBa

global best_acc
best_acc=100

def save_checkpoint(state, is_best, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
 

def train(figures, labels, model, opt, Stock_name, use_cuda):
    opt=opt
    best_acc=100
    model = model
    optimize = optim.Adam(model.parameters(), lr = opt.lr, betas = (0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    title = 'CANDLE'
    batch = opt.batch
    train_ratio = opt.train_ratio
    train_num = int(train_ratio*len(labels))
    print(figures[:train_num].shape)
    print(figures[train_num:].shape)
    
    means=[0.485,0.456,0.406]
    stds = [0.229,0.224,0.225]
    train = figures[:train_num]
    """
    for i in range(3):
        means[i] = np.mean(train[:,:,:,i])
        stds[i]=np.std(train[:,:,:,i])
    
    print(means)
    print(stds)
    """
    
    
    train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
                            tv.transforms.Resize(dimension),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean = means, std = stds)
                            ])

    valid_transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
                            tv.transforms.Resize(dimension),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean =means, std = stds)
                            ])
    
    trainset = Pathdataset(image = figures[:train_num], labels = labels[:train_num],test_mode = False, transform=train_transform)
    validset = Pathdataset(image = figures[train_num:], labels = labels[train_num:],test_mode = False, transform=valid_transform)
    
    trainloader = DataLoader(dataset=trainset, batch_size=batch, shuffle=True, drop_last = True)
    validloader = DataLoader(dataset=validset, batch_size=batch, shuffle=True, drop_last = True)
    
    if opt.resume:
        print('Resuming from checkpoint')
        assert os.path.isfile(opt.resume),'Error: no checkpoint dir'
        opt.checkpoint = os.path.dirname(opt.resume)
        
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title = title,resume = True)
    else:
        os.mkdir(opt.checkpoint+'_'+Stock_name)
        
        logger = Logger(os.path.join(opt.checkpoint+'_'+Stock_name, 'log.txt'), title= title)
        logger.set_names(['Train_loss','Valid_loss','Valid_acc'])
    
    for epoch in tqdm(range(opt.start_epoch, opt.finish_epoch)):
        
        losses = AVERAGEMETER()
        vlosses=AVERAGEMETER()
        vacces = AVERAGEMETER()

        
        
        model.train()  
        for batch_idx, (inputs, targets) in enumerate(trainloader):

                      
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            train_result = model(inputs)
            loss = criterion(train_result, targets)

            optimize.zero_grad()

            loss.backward(retain_graph = True)
            optimize.step()

            #print(str(np.max(train_result.cpu().detach().numpy(),axis=1)))
            losses.update(loss.item(), inputs.size(0))

        model.eval()
        for batch_idx, (inputs, targets) in enumerate(validloader):
            
            
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                valid_result = model(inputs)
            vloss = criterion(valid_result, targets)
            valid_result=valid_result.cpu().detach().numpy()
            #print(str(valid_result))
            valid_result = np.argmax(valid_result,axis=1)

            vacc = accuracy_score(valid_result, targets.cpu().detach().numpy())
            
            #print('predicted '+str(np.count_nonzero(valid_result)))
            #print('answer '+str(np.count_nonzero(targets.cpu().detach().numpy())))
            vacces.update(vacc,1)
            vlosses.update(vloss.item(),inputs.size(0))
        
        train_loss= losses.avg
        valid_loss= vlosses.avg
        valid_acc=vacces.avg
        
        is_best = valid_loss<=best_acc
        best_acc = min(valid_acc, best_acc)
        
        logger.append([train_loss, valid_loss ,valid_acc])
        if is_best==True:

            save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':model.state_dict(),
                    'loss':valid_loss,
                    'best_acc':best_acc
                    }, is_best, checkpoint=opt.checkpoint+'_'+Stock_name, filename = 'model_cnn.pth.tar')
        print('\nEpoch: [%d | %d] train_loss: %f valid_loss: %f valid_acc: %f' % (epoch, opt.finish_epoch,train_loss,valid_loss, valid_acc))
    logger.close()
    logger.plot()
    savefig(os.path.join(opt.checkpoint+'_'+Stock_name,'log.png'))

    print('Finish Best is: '+ str(best_acc))

def cnn_predict(figures, opt, Stock_name):
    opt = opt
    batch = 1
    means=[0.485,0.456,0.406]
    stds = [0.229,0.224,0.225]
    os.environ['CUDA_VISIBLE_DEVICES']=opt.device
    use_cuda = torch.cuda.is_available()
    cudnn.benchmark=True

    model = EfficientNet.from_name(model_name='efficientnet-b3', num_classes = 2, image_size=244).cuda()
    pretrain = torch.load(os.path.join(checkpoint+'_'+Stock_name,'model_cnn.pth.tar'), map_location=torch.device('cuda'))
    model.load_state_dict(pretrain['state_dict'])
    
    test_transform = tv.transforms.Compose([tv.transforms.ToPILImage(mode = 'RGB'),
                        tv.transforms.Resize(dimension),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean =means, std = stds)
                        ])

    testset = Pathdataset(image = figures,test_mode = True, transform=test_transform)
    testloader = DataLoader(dataset=testset, batch_size=batch, shuffle=False, drop_last = False)
    model.eval()
    for batch_idx, inputs in enumerate(testloader):
        
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():

            result = model(inputs)
        print(result)
        result = result.cpu()
        _, pred = result.topk(1, 1, True, True)
        pred= pred.view(-1,1)
        pred=pred.detach().numpy()
        print(pred)
    return pred[0][0]


def train_for_FB(Stock_price,Stock_name,labels, opt):
    Stock_price = fdr.DataReader(Stock_name, opt.start_date)
    opt=opt
    best_acc=0

    train_ratio = opt.train_ratio
    train_num = int(train_ratio*len(labels))
    delta_days = (Stock_price.index[-1] - Stock_price.index[train_num-1]).days 
    totalset = FBdataset(Stock_price)
    trainset = FBdataset(Stock_price[:train_num])
    validset = FBdataset(Stock_price[train_num:])	
    
    parameters = pd.DataFrame(columns=['mse','parameter'])
    forecast_list=[]
    params_list=[]
    params_grid = { 'changepoint_prior_scale':[0.05,0.1,0.15],
                    'changepoint_range':[0.7,0.8,0.9]} 
    grid = ParameterGrid(params_grid)
    
    for p in grid:
        print( '************Parameter :' +str(p) +'************')
        m =Prophet(**p)
        m.fit(trainset)
        future = m.make_future_dataframe(periods=delta_days)   
        future['cap'] = trainset.y.max()
        future['floor'] = trainset.y.min()
        forecast = m.predict(future)
        forecast = forecast[['ds','yhat']]
        #forecast.columns = ['Date','yhat']
        #forecast.set_index('Date',inplace=True)
        forecast_list.append(forecast)
        params_list.append(p)

    for k in range(len(forecast_list)):
        pred = pd.merge(validset,forecast_list[k],how='left',on='ds')
        pred = pred.fillna(0)
        mse = mean_squared_error(pred['y'],pred['yhat'])
        parameters = parameters.append({'mse':mse,'parameter':params_list[k]},ignore_index=True) 

    best_acc=parameters[parameters['mse']==parameters['mse'].min()]['mse']
    best_acc_index=parameters.index[parameters['mse']==best_acc.values[0]]
    best_df=forecast_list[best_acc_index[0]][['ds','yhat']]
    
    
    m =Prophet(**(parameters.loc[best_acc_index[0]]['parameter']))
    m.fit(totalset)
    """
    if os.path.exists(opt.checkpoint+'_'+Stock_name):
        pass
    else:
        os.mkdir(opt.checkpoint+'_'+Stock_name)
    """    
    filename = os.path.join(opt.checkpoint+'_'+Stock_name,"checkpoint_FB.pkl")
    with open(filename, "wb") as f:
            pickle.dump((m,best_df), f)
    #parameters[best_acc_index[0]]['parameter'].to_pickle(filename)

    print(best_df)      
    print('Finish Best is: '+ str(best_acc.values[0]))
    
    """
    future = m.make_future_dataframe(periods=delta_days+1, freq='d') 
    future['cap'] = trainset.y.max()
    future['floor'] = trainset.y.min()

    forecast = m.predict(future)
    #forecast = pd.merge(test_set, forecast, how='left', on='ds') 
    forecast = forecast[['ds','yhat']]
    print(forecast)    
    return forecast['yhat'].tail(1)
    """



def predict_for_FB(opt, Stock_name):   
    opt = opt
    filename = os.path.join(opt.checkpoint+'_'+Stock_name,"checkpoint_FB.pkl")
    with open(filename, 'rb') as f:
        m, best_df = pickle.load(f)
    day = (best_df.iloc[-1,0]).strftime("%Y-%m-%d")
    print(day)
    test_set = FBdataset(fdr.DataReader(Stock_name, day))
    #test_set = test_set.append ({'ds':datetime.now()+timedelta(days=1), 'y':0,'cap':0,'floor':0},ignore_index=True)
    delta_days = (datetime.now()- datetime.strptime(day, "%Y-%m-%d")).days
    future = m.make_future_dataframe(periods=delta_days+1, freq='d') 
    future['cap'] = test_set.y.max()
    future['floor'] = test_set.y.min()
   
    forecast = m.predict(future)
    #forecast = pd.merge(test_set, forecast, how='left', on='ds') 
    forecast = forecast[['ds','yhat']]
    print(forecast)    
    return forecast['yhat'].tail(1)


    
def main(Stock_name, Stock_price, figures, labels,opt):
    Stock_name = Stock_name
    os.environ['CUDA_VISIBLE_DEVICES']=opt.device
    use_cuda = torch.cuda.is_available()
    if opt.manualSeed is None:
        opt.manualSeed=random.randint(1,10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    cudnn.benchmark = True
    """
    tmp_data = next(train_gen)
    print("Chart image shape : ",np.shape(tmp_data[0]))
    print("Label shape :",np.shape(tmp_data[1]))
    """
    
    model=EfficientNet.from_pretrained('efficientnet-b3',num_classes=2, image_size = 244).cuda()

    # transfer learning
    for n, p in model.named_parameters():
        if '_fc' not in n:
            p.requires_grad = False
        else:
            p.requires_grad=True

    print('model efficientnet')
    train(figures, labels, model, opt, Stock_name, use_cuda)

    # model = models.FBProphet()
    print('model FBProphet')
    train_for_FB(Stock_price,Stock_name, labels, opt) # model
        




    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type = int, default=0)
    parser.add_argument('--finish_epoch', type = int, default=50)
    parser.add_argument('--batch', type = int, default=20)
    
    parser.add_argument('--device', type = str, default='0')
    
    parser.add_argument('--manualSeed', type = int, default=None)

    
    #seq_len, dimension, start_date,manualSeed,select, train_ratio, resume=str, model=str
    parser.add_argument('--add_v', type = int, default=0)    
    parser.add_argument('--start_date', type = str, default='2000-01-01')
    parser.add_argument('--train_ratio', type = float, default=0.9)
    parser.add_argument('--resume', type = str)

    parser.add_argument('--lr', type = float, default=0.0001)
    parser.add_argument('--checkpoint', type = str, default='checkpoint')
    

    opt=parser.parse_args()
    
    checkpoint = opt.checkpoint
    code_name = fdr.StockListing('KRX')
    
    try:
        DB = DBa.database()
        print('DB connection')
    except:
        print('DB connection failed')
        exit()
    
    print('start train')
  


    try:
        df_result = pre_data.naver()
    except:
        print('re-connect naver finance')
        df_result = pre_data.naver()
   
    DB=DBa.database()
    DB.reset('CRAWL_STOCK')
    
    data=[]
    for i in range(len(df_result.columns)):
        for j in range(len(df_result)):
            try:
                
                name = df_result[df_result.columns[i]][j]
                code = code_name['Symbol'][code_name['Name']==name]
                print(code.values[0])
                if i<5:
                    k=[code.values[0],i]
                else:
                    k=[code.values[0],i-5]
                data.append(k)
            except:
                continue

    DB.insert(data)
    
    DB.reset('VIEW_INTEREST')
    DB.reset('VIEW_RETAINED')
    DB.copy_table()
    
    retained = DB.retained_get()
    interest = DB.interest_get()


    interest_list = set()
    retained_list=set()
    crawl_list = set()
    for i in range(len(data)):
        crawl_list.add(data[i][0])
    for i in range(len(interest)):
        interest_list.add(interest[i][1])
    for i in range(len(retained)):
        retained_list.add(retained[i][1])
    print('first listing complete!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #############################################################################

    dimension = 244
    seq_len = 20
    period = 20
    pb=2
    start_date = opt.start_date
    
    crawl_top_list = pre_data.bollinger(period=period,pb=2,pre=1,min_per=1,start_date = start_date,df_result=crawl_list)
    retained_top_list= pre_data.bollinger(period=period,pb=2,pre=1,min_per=1,start_date = start_date,df_result=retained_list)
    interest_top_list = pre_data.bollinger(period=period,pb=2,pre=1,min_per=1,start_date = start_date,df_result=interest_list)



    crawl_Sale_com, crawl_Buy_com = pre_data.second_check(crawl_top_list, crawl_list)
    retained_Sale_com, retained_Buy_com = pre_data.second_check(retained_top_list, retained_list)
    interest_Sale_com, interest_Buy_com = pre_data.second_check(interest_top_list, interest_list)
    
    Buy_com=set()
    Sale_com=set()
    for i in range(len(crawl_Buy_com)):
        Buy_com.add(crawl_Buy_com[i])
    for i in range(len(retained_Buy_com)):
        Buy_com.add(retained_Buy_com[i])
    for i in range(len(interest_Buy_com)):
        Buy_com.add(interest_Buy_com[i])

    
    for i in range(len(retained_Sale_com)):
        Sale_com.add(retained_Sale_com[i])
    
    Buy_com = list(Buy_com)
    Sale_com = list(Sale_com)
    
    print(Buy_com)
    print(Sale_com)

    
    ##########################

    dimension = 244
    seq_len = 20
    period = 20
    pb=2
    start_date = opt.start_date
    buy_data=[]
    sell_data=[]
    
    

    Buy_com = ['020120', '072520', '257370', '068270', '091440', '139670', '134060', '227100', '057880', '015260', '08537M', '086900']

    # Sale_com =['072520']


    for i in range(len(Buy_com)):   
        
        selected_com = Buy_com[i]
        stock_price = fdr.DataReader(selected_com, start_date)
        if int(len(stock_price)*opt.train_ratio)<=30:
            opt.batch = 1
        elif int(len(stock_price)*opt.train_ratio)<=60:
            opt.batch = 10
        #exist pretrained file
        if checkpoint+'_'+selected_com in os.listdir('/home/ubuntu/Candle_aws'):
            figures = print_Candle.test_ohlc2cs(df = stock_price, dimension = dimension, seq_len = seq_len, add_v = opt.add_v, stock_name = selected_com)
            cnn_result = cnn_predict(figures, opt, selected_com)
            FB_result=predict_for_FB(opt, selected_com)
            #FB_result = train_for_FB(stock_price,selected_com, stock_price, opt)
            
        #not exist pretrained file
        else:

            #selected_code = top['Symbol'][top['Name']==selected_com]
            #stock_price = fdr.DataReader('010140', start_date)
            figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len, add_v = opt.add_v)

            print(np.shape(labels), np.shape(figures))
            main(selected_com, stock_price ,figures, labels,opt)
            figures = print_Candle.test_ohlc2cs(df = stock_price, dimension = dimension, seq_len = seq_len, add_v = opt.add_v, stock_name = selected_com)
            cnn_result = cnn_predict(figures, opt, selected_com)
            FB_result=predict_for_FB(opt, selected_com)
        res_crawl = DB.check_crawl(selected_com)
        
        
        if stock_price['Close'].tail(1).values[0] > float(FB_result):
            FB_bin = 0
        else:
            FB_bin=1
        bol_bin = 1
        
        if cnn_result==1 and FB_bin ==1:
            up_down = 2
        elif cnn_result != FB_bin:
            up_down = 0
        else:
            up_down = 1
        
        
        FB_percent = 100*(float(FB_result)-stock_price['Close'].tail(1).values[0])/stock_price['Close'].tail(1).values[0]
        print(str(FB_percent))
        if FB_percent > 30:
            FB_percent =30.000
        elif FB_percent<-30:
            FB_percent = -30.000
        if len(res_crawl)!=0:
            for k in range(len(res_crawl)):
                
                data = [selected_com,round(float(FB_percent),2),int(cnn_result),int(up_down),int(res_crawl[k][0])]
                buy_data.append(data)
        else:
            data = [selected_com,round(float(FB_percent),2),int(cnn_result),int(up_down),5]
            buy_data.append(data)


    DB.reset('BUY_LIST')
    DB.insert_buylist(buy_data)
        

    for i in range(len(Sale_com)):   
        
        selected_com = Sale_com[i]
        stock_price = fdr.DataReader(selected_com, start_date)
        if int(len(stock_price)*opt.train_ratio)<=30:
            opt.batch = 1
        elif int(len(stock_price)*opt.train_ratio)<=60:
            opt.batch = 20

        if checkpoint+'_'+selected_com in os.listdir('/home/ubuntu/Candle_aws'):
            figures = print_Candle.test_ohlc2cs(df = stock_price, dimension = dimension, seq_len = seq_len, add_v = opt.add_v, stock_name = selected_com)
            cnn_result = cnn_predict(figures, opt, selected_com)
            FB_result=predict_for_FB(opt, selected_com)
            # FB_result = train_for_FB(stock_price,selected_com, stock_price, opt)

        else:    
            #selected_code = top['Symbol'][top['Name']==selected_com]
            #stock_price = fdr.DataReader('010140', start_date)
            figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len, add_v = opt.add_v)

            print(np.shape(labels), np.shape(figures))
        
            main(selected_com, stock_price ,figures, labels,opt)
            figures = print_Candle.test_ohlc2cs(df = stock_price, dimension = dimension, seq_len = seq_len, add_v = opt.add_v, stock_name = selected_com)
            cnn_result = cnn_predict(figures, opt, selected_com)
            FB_result=predict_for_FB(opt, selected_com)

        if stock_price['Close'].tail(1).values[0] > float(FB_result):
            FB_bin = 0
        else:
            FB_bin=1
        bol_bin = 0
        
        if cnn_result==0 and FB_bin ==0:
            up_down = 2
        elif cnn_result != FB_bin:
            up_down = 0
        else:
            up_down = 1

        FB_percent = 100*(float(FB_result)-stock_price['Close'].tail(1).values[0])/stock_price['Close'].tail(1).values[0]
        if FB_percent > 30:
            FB_percent =30.000
        elif FB_percent<-30:
            FB_percent = -30.000

        data = [selected_com,round(float(FB_percent),2),int(cnn_result),int(up_down)]
        sell_data.append(data)

    DB.reset('SELL_LIST')
    DB.insert_selllist(sell_data)



#######################


    """
    stock_li=['010140','010140','019170','003000']
    labels_final = []
    for i in range(4):   
        print(str(i))
        selected_com = "test"
        # selected_code = top['Symbol'][top['Name']==selected_com]
        #stock_price = fdr.DataReader(selected_code.values[0], start_date)
        stock_price = fdr.DataReader(stock_li[i], start_date)
        figures, labels = print_Candle.ohlc2cs(df=stock_price, dimension=dimension, seq_len=seq_len, add_v = opt.add_v)
        if i ==0:
            figures_final = figures
        else:

            figures_final = np.concatenate((figures_final, figures),axis=0)
        labels_final.extend(labels)

    print(np.shape(labels_final), np.shape(figures_final))
        
    main(selected_com, stock_price ,figures_final, labels_final)
    """
        
    

#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets



args = {}
args['lr'] = 0.001
args['epochs'] = 256
args['batch-size'] = 64
args['log_interval'] = 100
args['test_interval'] = 100
args['save_interval'] =500
args['save_dir'] ='snapshot'
args['early_stop'] = 100
args['save_best'] = True
args['shuffle'] =False
args['dropout'] = 0.5
args['max-norm'] = 3
args['embed_dim'] = 128
args['kernel_num'] = 100
args['kernel-sizes'] = '3,4,5'
args['static'] = False
args['device'] = -1
args['no_cuda'] = None
args['snapshot'] = None
args['predict'] = None
args['test'] = False
args['kmax_pooling'] = 1
args['hidden_dim'] = 100
args['lstm_layers'] = 1
args['linear_hidden_size'] = 100
args['gru_layers'] =1

args['rcnn_kernel'] =512

# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args['batch-size'],
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args['batch-size'], len(dev_data)),
                                **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args['embed_num'] = len(text_field.vocab)
args['class_num'] = len(label_field.vocab) - 1
args['cuda'] = (not args['no_cuda']) and torch.cuda.is_available(); del args['no_cuda']
args['kernel_sizes'] = [int(k) for k in args['kernel-sizes'].split(',')]
args['save_dir'] = os.path.join(args['save_dir'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))
for key,value in args.items():
    print(key,' : ',value)



# model
from model import TextCNN,GRU,AttLSTM,bigru_attention,RCNN,GRU_CNN
# cnn = model.CNN_Text(args)
# cnn = TextCNN(args['embed_num'],args['embed_dim'],args['kernel_num'],50 ,100,args['class_num'])
# cnn = GRU(args)
# cnn = AttLSTM(args)
# cnn = bigru_attention(args)
# cnn = RCNN(args)
cnn = GRU_CNN(args)




if args['snapshot'] is not None:
    print('\nLoading model from {}...'.format(args['snapshot']))
    cnn.load_state_dict(torch.load(args['snapshot']))

if args['cuda']:
    torch.cuda.set_device(args['device'])
    cnn = cnn.cuda()
        

# train or predict
if args['predict'] is not None:
    label = train.predict(args['predict'], cnn, text_field, label_field, args['cuda'])
    print('\n[Text]  {}\n[Label] {}\n'.format(args['predict'], label))
elif args['test']:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

#
#
for batch in train_iter:
    feature, target = batch.text, batch.label
    break
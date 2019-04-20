import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        Ci = 1
        Co = args['kernel_num']
        Ks = args['kernel_sizes']

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args['dropout'])
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.args['static']:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit



kernal_sizes = [ 3, 4,5]


class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,kernel_num,max_text_len ,linear_hidden_size,label_size,vectors=None):
        super(TextCNN, self).__init__()

        '''Embedding Layer'''
        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=kernel_num,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(kernel_num),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=max_text_len - kernel_size+1),

                # nn.Conv1d(in_channels=kernel_num,
                #           out_channels=kernel_num,
                #           kernel_size=kernel_size),
                # nn.BatchNorm1d(kernel_num),
                # nn.ReLU(inplace=True),
                # nn.MaxPool1d(kernel_size=(max_text_len - kernel_size*2 + 2))
            )
            for kernel_size in kernal_sizes
        ]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(3 * kernel_num, label_size)

            # nn.Linear(3 * kernel_num, linear_hidden_size),
            # nn.BatchNorm1d(linear_hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(linear_hidden_size, label_size)
        )

    def forward(self, inputs):
        embeds = self.embedding(inputs)###inputs的shape=B*max_len*em_dim
        ####embed shape=B*max_len*embed_dim
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out = [conv(embeds.permute(0,2,1)) for conv in self.convs]#######conv_out[i] shape = B*embed_dim*1
        conv_out = torch.cat(conv_out, dim=1)

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits

##############---------------------------------------------------------------------------------------
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]  # torch.Tensor.topk()的输出有两项，后一项为索引
    return x.gather(dim, index)

class GRU(nn.Module):
    def __init__(self, config, vectors=None):
        super(GRU, self).__init__()
        self.opt = config
        self.kmax_pooling = config['kmax_pooling']

        # GRU
        self.embedding = nn.Embedding(config['embed_num'], config['embed_dim'])
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.bigru = nn.GRU(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=config['dropout'],
            bidirectional=True)

        # self.fc = nn.Linear(args.hidden_dim * 2 * 2, args.label_size)
        # 两层全连接层，中间添加批标准化层
        # 全连接层隐藏元个数需要再做修改
        self.fc = nn.Sequential(
            nn.Linear(self.kmax_pooling * (config['hidden_dim'] * 2), config['linear_hidden_size']),
            nn.BatchNorm1d(config['linear_hidden_size']),
            nn.ReLU(inplace=True),
            nn.Linear(config['linear_hidden_size'], config['class_num'])
        )

    # 对LSTM所有隐含层的输出做kmax pooling
    def forward(self, text):
        embed = self.embedding(text)  #  nputs的shape=B*max_len
        ####embed shape=B*max_len*em_dim
        ####bigur[0]的shape= B*max_len*layers*hiddensize
        out = self.bigru(embed)[0].permute(0, 2, 1)  # batch * layers*hidden * seq
        pooling = kmax_pooling(out, 2, self.kmax_pooling)  # batch * hidden * kmax

        # word+article
        flatten = pooling.view(pooling.size(0), -1)
        logits = self.fc(flatten)

        return logits



class AttLSTM(nn.Module):
    def __init__(self, args, vectors=None):
        super(AttLSTM, self).__init__()
        self.args = args

        # LSTM
        self.embedding = nn.Embedding(args['embed_num'], args['embed_dim'])
        # self.embedding.weight.data.copy_(vectors)
        self.bilstm = nn.LSTM(
            input_size=args['embed_dim'],
            hidden_size=args['hidden_dim'],
            num_layers=args['lstm_layers'],
            batch_first=False,
            dropout=args['dropout'],
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(args['hidden_dim']*2, args['linear_hidden_size']),
            nn.BatchNorm1d(args['linear_hidden_size']),
            nn.ReLU(inplace=True),
            nn.Linear(args['linear_hidden_size'], args['class_num'])
        )

    def attention(self, rnn_out, state,hidden_num):
        merged_state = torch.cat([s for s in state], 1)
        # merged_state = merged_state[:,:hidden_num*self.args['lstm_layers']]+merged_state[:,self.args['lstm_layers']*hidden_num:]
        # out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        merged_state = merged_state.unsqueeze(2)
        # (batch, seq, hidden) * (batch, hidden, 1) = (batch, seq, 1)
        # print('merge_state shape:',merged_state.shape)
        # print('rnn out shape:',rnn_out.shape)
        weights = torch.bmm(rnn_out.permute(0, 2, 1), merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        # (batch, hidden, seq) * (batch, seq, 1) = (batch, hidden, 1)
        return torch.bmm(rnn_out, weights).squeeze(2)


    def forward(self, text):
        text.data.t_()  ########输入的text为bantch*max_len
        embed = self.embedding(text)  # seq * batch * emb
        out, hidden = self.bilstm(embed)
        out = out.permute(1, 2, 0)  # batch * hidden * seq
        h_n, c_n = hidden
        att_out = self.attention(out, h_n,self.args['hidden_dim'])

        logits = self.fc(att_out)

        return logits


class bigru_attention(nn.Module):
    def __init__(self, args, vectors=None):
        self.args = args
        super(bigru_attention, self).__init__()
        self.hidden_dim = args['hidden_dim']
        self.gru_layers = args['gru_layers']

        self.embedding = nn.Embedding(args['embed_num'], args['embed_dim'])
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        self.bigru = nn.GRU( args['embed_dim'], self.hidden_dim // 2, num_layers=self.gru_layers, bidirectional=True)
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.fc = nn.Linear(self.hidden_dim, args['class_num'])

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, sentence):
        sentence.data.t_()
        embeds = self.embedding(sentence)  # [seq_len, bs, emb_dim]
        gru_out, _ = self.bigru(embeds)  # [seq_len, bs, hid_dim]
        x = gru_out.permute(1, 0, 2)
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = torch.nn.functional.softmax(att, dim=1)
        scored_x = x * att_score
        feat = torch.sum(scored_x, dim=1)
        y = self.fc(feat)
        return y




class RCNN(nn.Module):
    def __init__(self, args, vectors=None):
        super(RCNN, self).__init__()
        self.kmax_k = args['kmax_pooling']
        self.config = args

        #
        self.embedding = nn.Embedding(args['embed_num'], args['embed_dim'])
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.lstm = nn.LSTM(
            input_size=args['embed_dim'],
            hidden_size=args['hidden_dim'],
            num_layers=args['lstm_layers'],
            batch_first=False,
            dropout=args['dropout'],
            bidirectional=True
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args['hidden_dim'] * 2 + args['embed_dim'], out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )

        # classifer
        # self.fc = nn.Linear(2 * (100 + 100), args.label_size)
        self.fc = nn.Sequential(
            nn.Linear(2 * 100, args['linear_hidden_size']),
            nn.BatchNorm1d(args['linear_hidden_size']),
            nn.ReLU(inplace=True),
            nn.Linear(args['linear_hidden_size'], args['class_num'])
        )

    def forward(self, text):
        text.data.t_()
        embed = self.embedding(text)#####len*batch*embed_dim
        out = self.lstm(embed)[0].permute(1, 2, 0) ####len*batch*(2*hidden)----->batch*(2*hidden)*len
        out = torch.cat((out, embed.permute(1, 2, 0)), dim=1)#####batch*(hideen*2+embed_dim)*len
        conv_out = kmax_pooling(self.conv(out), 2, self.kmax_k)#####

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits



class GRU_CNN(nn.Module):
    def __init__(self, args, vectors=None):
        super(GRU_CNN, self).__init__()
        self.kmax_k = args['kmax_pooling']
        self.config = args

        #
        self.embedding = nn.Embedding(args['embed_num'], args['embed_dim'])
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.gru = nn.GRU(
            input_size=args['embed_dim'],
            hidden_size=args['hidden_dim'],
            num_layers=args['lstm_layers'],
            batch_first=False,
            dropout=args['dropout'],
            bidirectional=True
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args['hidden_dim'] * 2 + args['embed_dim'], out_channels=args['rcnn_kernel'], kernel_size=3),
            nn.BatchNorm1d(args['rcnn_kernel']),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=args['rcnn_kernel'], out_channels=args['rcnn_kernel'], kernel_size=3),
            nn.BatchNorm1d(args['rcnn_kernel']),
            nn.ReLU(inplace=True)
        )

        # classifer
        # self.fc = nn.Linear(2 * (100 + 100), args.label_size)
        self.fc = nn.Sequential(
            nn.Linear(args['kmax_pooling'] * args['rcnn_kernel'], args['linear_hidden_size']),
            nn.BatchNorm1d(args['linear_hidden_size']),
            nn.ReLU(inplace=True),
            nn.Linear(args['linear_hidden_size'],  args['class_num'])
        )

    def forward(self, text):
        text.data.t_()
        embed = self.embedding(text)#######len*batch*embe_dim
        out = self.gru(embed)[0].permute(1, 2, 0)####len*batch*(2*hidden_dim)----->Batch*(2*hidden)*len
        out = torch.cat((out, embed.permute(1, 2, 0)), dim=1)
        conv_out = kmax_pooling(self.conv(out), 2, self.kmax_k)

        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits

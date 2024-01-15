import collections
import logging
import pickle
import torch
from torch.nn import BCEWithLogitsLoss, Dropout, Linear
from transformers import AutoModel, XLNetModel
import torch.nn as nn
import torch.nn.functional as F
from models.utils import initial_code_title_vectors
from torch.nn.parameter import Parameter
from math import floor
import math
import numpy as np

logger = logging.getLogger("lwat")
newwikivec = np.load(r"../newwikivec.npy", allow_pickle=True)
wikivec = torch.FloatTensor(newwikivec).to('cuda:0')  # wikipedia_konwledge
co_icd_matrix = pickle.load(open(r'../drug.pkl', "rb"))
drug = co_icd_matrix.values
icd_drug = torch.FloatTensor(drug).to('cuda:0')
# icd_drug = F.normalize(icd_drug, dim=-1)



class CodingModelConfig:
    def __init__(self,
                 transformer_model_name_or_path,
                 transformer_tokenizer_name,
                 transformer_layer_update_strategy,
                 num_chunks,
                 max_seq_length,
                 dropout,
                 dropout_att,
                 d_model, # hidden_size
                 label_dictionary,
                 num_labels,
                 use_code_representation,
                 code_max_seq_length,
                 code_batch_size,
                 multi_head_att,
                 chunk_att,
                 linear_init_mean,
                 linear_init_std,
                 document_pooling_strategy,
                 multi_head_chunk_attention):
        super(CodingModelConfig, self).__init__()
        self.transformer_model_name_or_path = transformer_model_name_or_path
        self.transformer_tokenizer_name = transformer_tokenizer_name
        self.transformer_layer_update_strategy = transformer_layer_update_strategy
        self.num_chunks = num_chunks
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.dropout_att = dropout_att
        self.d_model = d_model
        # labels_dictionary is a dataframe with columns: icd9_code, long_title
        self.label_dictionary = label_dictionary
        self.num_labels = num_labels
        self.use_code_representation = use_code_representation
        self.code_max_seq_length = code_max_seq_length
        self.code_batch_size = code_batch_size
        self.multi_head_att = multi_head_att
        self.chunk_att = chunk_att
        self.linear_init_mean = linear_init_mean
        self.linear_init_std = linear_init_std
        self.document_pooling_strategy = document_pooling_strategy
        self.multi_head_chunk_attention = multi_head_chunk_attention
        
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    # self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
    device = torch.device('cuda')
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.adj = adj
        self.w = adj.shape[0]
        # 假设 adj 是一个 DataFrame
        # 检查 adj 的类型和形状
        # print(type(adj))  # 它应该是 torch.Tensor
        # print(adj.shape)  # 确保它具有所需的形状

        adj = adj + torch.eye(voc_size).to('cuda:0')

        # adj = self.normalize(adj + torch.eye(50).to(adj.device))
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, 768)  ## graph_transformer 改为16

    def forward(self,adj):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        edge = self.adj

        return node_embedding
    
# class ContextAwareBiGate(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ContextAwareBiGate, self).__init__()
#         self.linear_forward = nn.Linear(input_size, output_size)
#         self.linear_backward = nn.Linear(input_size, output_size)
#         self.context_attention = nn.Linear(input_size, 1)

#     def forward(self, input, context):
#         forward_gate = torch.sigmoid(self.linear_forward(input)) #前向门 状态门
#         backward_gate = torch.sigmoid(self.linear_backward(input)) # 后向门 信息门

#         context_weights = F.softmax(self.context_attention(context), dim=1)
#         context_forward = torch.matmul(context_weights.transpose(1, 2), input)
#         context_backward = torch.matmul(context_weights.flip(dims=[2]).transpose(1, 2), input)

#         output = input * forward_gate + context_backward * backward_gate
#         return output    
    
class LableWiseAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(LableWiseAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model,
                                         self.config.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.config.d_model, self.config.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        # Mean pooling last hidden state of code title from transformer model as the initial code vectors
        self._init_linear_weights(mean=self.config.linear_init_mean, std=self.config.linear_init_std)

    def _init_linear_weights(self, mean, std):
        # normalize the l1 weights
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        if self.config.use_code_representation:
            code_vectors = initial_code_title_vectors(self.config.label_dictionary,
                                                      self.config.transformer_model_name_or_path,
                                                      self.config.transformer_tokenizer_name
                                                      if self.config.transformer_tokenizer_name
                                                      else self.config.transformer_model_name_or_path,
                                                      self.config.code_max_seq_length,
                                                      self.config.code_batch_size,
                                                      self.config.d_model,
                                                      self.args.device)

            self.l2_linear.weight = torch.nn.Parameter(code_vectors, requires_grad=True)
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

    def forward(self, x):
        # input: (batch_size, max_seq_length, transformer_hidden_size)
        # output: (batch_size, max_seq_length, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))
        # softmax(UZ)
        # l2_linear output shape: (batch_size, max_seq_length, num_labels)
        # attention_weight shape: (batch_size, num_labels, max_seq_length)
        attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
        # attention_output shpae: (batch_size, num_labels, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight
    
class DrugWiseAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(DrugWiseAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model,
                                         self.config.num_labels, bias=False) #(768,50)weight就是转置
        self.tanh = torch.nn.Tanh()
#         self.l2_linear = torch.nn.Linear(self.config.d_model, self.config.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drug_icd_gcn = GCN(voc_size=50, emb_dim=self.config.d_model, adj=icd_drug)

        # Mean pooling last hidden state of code title from transformer model as the initial code vectors
#         self._init_linear_weights(mean=self.config.linear_init_mean, std=self.config.linear_init_std)

    def forward(self, x): # [2, 512, 768]----[2, 50, 768]
        # input: (batch_size, max_seq_length, transformer_hidden_size)
        # output: (batch_size, max_seq_length, transformer_hidden_size)? (batch_size, num_labels, transformer_hidden_size)
        # Z = Tan(WH) l1_weight的权重是 torch.Size([768, 768]),l1_output的尺寸是 torch.Size([2, 512, 768])
        drug_icd_embeddings = self.drug_icd_gcn(icd_drug)
#         print("drug_icd_embeddings的大小是",drug_icd_embeddings.shape) #(50,768)
        l1_output = self.tanh(x)
#         print("l1_weight的权重是",self.l1_linear.weight.shape)
#         print("l1_output的尺寸是",l1_output.shape)
        # softmax(UZ)
        # l2_linear output shape: (batch_size, max_seq_length, num_labels)
        # attention_weight shape: (batch_size, num_labels, max_seq_length)
        attention_weight = self.softmax((l1_output.matmul(drug_icd_embeddings.transpose(0,1)))).transpose(1, 2)
        # attention_output shpae: (batch_size, num_labels, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight
    
class WikiWiseAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(WikiWiseAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model,
                                         self.config.num_labels, bias=False)
        self.wiki1=torch.nn.Linear(314,50)
        self.wiki2=torch.nn.Linear(12173,768)

        self.tanh = torch.nn.Tanh()
#         self.l2_linear = torch.nn.Linear(self.config.d_model, self.config.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
#         self.drug_icd_gcn = GCN(voc_size=50, emb_dim=self.config.d_model, adj=icd_drug)

        # Mean pooling last hidden state of code title from transformer model as the initial code vectors
#         self._init_linear_weights(mean=self.config.linear_init_mean, std=self.config.linear_init_std)

    def forward(self, x): # [2, 512, 768]----[2, 50, 768]
        # input: (batch_size, max_seq_length, transformer_hidden_size)
        # output: (batch_size, max_seq_length, transformer_hidden_size)? (batch_size, num_labels, transformer_hidden_size)
        # Z = Tan(WH) l1_weight的权重是 torch.Size([768, 768]),l1_output的尺寸是 torch.Size([2, 512, 768])

        l1_output = self.tanh(self.wiki2(wikivec))
        l1_output = self.tanh(self.wiki1(l1_output.transpose(0, 1)))


        # softmax(UZ)
        # l2_linear output shape: (batch_size, max_seq_length, num_labels)
        # attention_weight shape: (batch_size, num_labels, max_seq_length)
        attention_weight = self.softmax(x.matmul(l1_output)).transpose(1, 2)
#         print("attention_weight",attention_weight.shape)
        # attention_output shpae: (batch_size, num_labels, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight


class ChunkAttentionLayer(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(ChunkAttentionLayer, self).__init__()

        self.config = coding_model_config
        self.args = args

        # layers
        self.l1_linear = torch.nn.Linear(self.config.d_model,
                                         self.config.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.config.d_model, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        self._init_linear_weights(mean=self.config.linear_init_mean, std=self.config.linear_init_std)

    def _init_linear_weights(self, mean, std):
        # initialize the l1
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

    def forward(self, x):
        # input: (batch_size, num_chunks, transformer_hidden_size)
        # output: (batch_size, num_chunks, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))
        # softmax(UZ)
        # l2_linear output shape: (batch_size, num_chunks, 1)
        # attention_weight shape: (batch_size, 1, num_chunks)
        attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
        # attention_output shpae: (batch_size, 1, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight

# define the model class
class CodingModel(torch.nn.Module):
    def __init__(self, coding_model_config, args):
        super(CodingModel, self).__init__()
        self.coding_model_config = coding_model_config
        self.args = args
        # layers
        self.transformer_layer = AutoModel.from_pretrained(self.coding_model_config.transformer_model_name_or_path)
        if isinstance(self.transformer_layer, XLNetModel):
            self.transformer_layer.config.use_mems_eval = False
        self.dropout = Dropout(p=self.coding_model_config.dropout)

        if self.coding_model_config.multi_head_att:
            # initial multi head attention according to the num_chunks
            self.label_wise_attention_layer = torch.nn.ModuleList(
                [LableWiseAttentionLayer(coding_model_config, args)
                 for _ in range(self.coding_model_config.num_chunks)])
        else:
            self.label_wise_attention_layer = LableWiseAttentionLayer(coding_model_config, args)
            self.drug_wise_attention_layer = DrugWiseAttentionLayer(coding_model_config, args)
            self.wiki_wise_attention_layer = WikiWiseAttentionLayer(coding_model_config, args)
        self.dropout_att = Dropout(p=self.coding_model_config.dropout_att)
        
        # 初始化GCN
#         self.drug_icd_gcn = GCN(voc_size=50, emb_dim=d_model, adj=self.adj_icd)
        # c初始化门控单元
        self.context_aware_bi_gate = ContextAwareBiGate(768, 768)
        self.weight1 = torch.nn.Linear(768, 1)
        self.weight2 = torch.nn.Linear(768, 1)
        self.wiki1 = torch.nn.Linear(768, 314)
        self.wiki1 = torch.nn.Linear(12, 1)
        
        # initial chunk attention
        if self.coding_model_config.chunk_att:
            if self.coding_model_config.multi_head_chunk_attention:
                self.chunk_attention_layer = torch.nn.ModuleList([ChunkAttentionLayer(coding_model_config, args)
                                                                  for _ in range(self.coding_model_config.num_labels)])
            else:
                self.chunk_attention_layer = ChunkAttentionLayer(coding_model_config, args)

            self.classifier_layer = Linear(self.coding_model_config.d_model,
                                           self.coding_model_config.num_labels)
        else:
            if self.coding_model_config.document_pooling_strategy == "flat":
                self.classifier_layer = Linear(self.coding_model_config.num_chunks * self.coding_model_config.d_model,
                                       self.coding_model_config.num_labels)
            else: # max or mean pooling
                self.classifier_layer = Linear(self.coding_model_config.d_model,
                                               self.coding_model_config.num_labels)
        self.sigmoid = torch.nn.Sigmoid()

        if self.coding_model_config.transformer_layer_update_strategy == "no":
            self.freeze_all_transformer_layers()
        elif self.coding_model_config.transformer_layer_update_strategy == "last":
            self.freeze_all_transformer_layers()
            self.unfreeze_transformer_last_layers()

        # initialize the weights of classifier
        self._init_linear_weights(mean=self.coding_model_config.linear_init_mean, std=self.coding_model_config.linear_init_std)

    def _init_linear_weights(self, mean, std):
        torch.nn.init.normal_(self.classifier_layer.weight, mean, std)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, targets=None):
        # input ids/mask/type_ids shape: (batch_size, num_chunks, max_seq_length)
        # labels shape: (batch_size, num_labels)
        transformer_output = []

        # pass chunk by chunk into transformer layer in the batches.
        # input (batch_size, sequence_length)
        for i in range(self.coding_model_config.num_chunks):
            l1_output = self.transformer_layer(input_ids=input_ids[:, i, :],
                                               attention_mask=attention_mask[:, i, :],
                                               token_type_ids=token_type_ids[:, i, :])
            # output hidden state shape: (batch_size, sequence_length, hidden_size)
            transformer_output.append(l1_output[0])

        # transpose back chunk and batch size dimensions
        transformer_output = torch.stack(transformer_output)
        transformer_output = transformer_output.transpose(0, 1)
        # dropout transformer output
        l2_dropout = self.dropout(transformer_output)
        # l2_dropout的维度是[2, 10, 512, 768]
        # Label-wise attention layers
        # output: (batch_size, num_chunks, num_labels, hidden_size)
        attention_output = []
        attention_weights = []

        for i in range(self.coding_model_config.num_chunks):
            # input: (batch_size, max_seq_length, transformer_hidden_size)
            if self.coding_model_config.multi_head_att:
                attention_layer = self.label_wise_attention_layer[i]
            else:
                attention_layer = self.label_wise_attention_layer
                drug_layer = self.drug_wise_attention_layer
                wiki_layer = self.wiki_wise_attention_layer
      
            l3_attention, attention_weight = attention_layer(l2_dropout[:, i, :])
            
            l3_drug_attention, drug_attention_weight = drug_layer(l2_dropout[:, i, :])
            l3_wiki_attention, wiki_attention_weight = wiki_layer(l2_dropout[:, i, :])
#             l3_drug_attention = torch.mul(l3_attention,l3_drug_attention)
            l3_drug_attention = torch.mul(l3_attention,l3_wiki_attention)

#             l3_attention = 0.5*l3_attention+ 0.5*l3_wiki_attention
#             l3_attention, attention_weight = drug_layer(l2_dropout[:, i, :]) 性能不好
#             weight1 = torch.sigmoid(self.weight1(l3_attention))
#             weight2 = torch.sigmoid(self.weight2(l3_drug_attention))
#             l3_attention = weight1 * l3_attention + weight2 * l3_drug_attention
#             l3_attention = 0.5*l3_attention+ 0.5*l3_drug_attention
#             l3_attention=self.context_aware_bi_gate(l3_attention, l3_drug_attention)#还可以
#             l3_attention=self.context_aware_bi_gate(l3_drug_attention, l3_attention) 性能不好
#             l3_attention=self.context_aware_bi_gate(l3_attention, l3_drug_attention)#还可以
#             l3_attention=self.context_aware_bi_gate(l3_attention, l3_wiki_attention)#还可以

#             print("加入成功",l3_drug_attention.shape)
#             print("经过标签注意力把输入为[2, 512, 768]的矩阵变为",l3_attention.shape=[2, 50, 768])
            # l3_attention shape: (batch_size, num_labels, hidden_size)
            # attention_weight: (batch_size, num_labels, max_seq_length)
            attention_output.append(l3_attention)
            attention_weights.append(attention_weight)

        attention_output = torch.stack(attention_output)
        attention_output = attention_output.transpose(0, 1)
        attention_weights = torch.stack(attention_weights)
        attention_weights = attention_weights.transpose(0, 1)
        
        # 所有attention的块stack后的大小l3_dropout torch.Size([2, 10, 50, 768])
        l3_dropout = self.dropout_att(attention_output)
#         print("所有attention的块stack的大小",l3_dropout.shape)

        if self.coding_model_config.chunk_att:
            # Chunk attention layers
            # output: (batch_size, num_labels, hidden_size)
            chunk_attention_output = []
            chunk_attention_weights = []

            for i in range(self.coding_model_config.num_labels):
                if self.coding_model_config.multi_head_chunk_attention:
                    chunk_attention = self.chunk_attention_layer[i]
                else:
                    chunk_attention = self.chunk_attention_layer
                l4_chunk_attention, l4_chunk_attention_weights = chunk_attention(l3_dropout[:, :, i]) # l4_chunk_attention[2, 1, 768]
                chunk_attention_output.append(l4_chunk_attention.squeeze(dim=1))
                chunk_attention_weights.append(l4_chunk_attention_weights.squeeze(dim=1))
                
            chunk_attention_output = torch.stack(chunk_attention_output) # chunk_attention_output[50, 2, 768]
            chunk_attention_output = chunk_attention_output.transpose(0, 1)
            chunk_attention_weights = torch.stack(chunk_attention_weights)
            chunk_attention_weights = chunk_attention_weights.transpose(0, 1)
            # output shape: (batch_size, num_labels, hidden_size)
            l4_dropout = self.dropout_att(chunk_attention_output)
        else:
            # output shape: (batch_size, num_labels, hidden_size*num_chunks)
            l4_dropout = l3_dropout.transpose(1, 2)
            if self.coding_model_config.document_pooling_strategy == "flat":
                # Flatten layer. concatenate representation by labels
                l4_dropout = torch.flatten(l4_dropout, start_dim=2)
            elif self.coding_model_config.document_pooling_strategy == "max":
                l4_dropout = torch.amax(l4_dropout, 2)
            elif self.coding_model_config.document_pooling_strategy == "mean":
                l4_dropout = torch.mean(l4_dropout, 2)
            else:
                raise ValueError("Not supported pooling strategy")

        # classifier layer
        # each code has a binary linear formula
        logits = self.classifier_layer.weight.mul(l4_dropout).sum(dim=2).add(self.classifier_layer.bias)

        preds = self.sigmoid(logits)

        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, targets)

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "label_attention_weights": attention_weights,
            "chunk_attention_weights": chunk_attention_weights if self.coding_model_config.chunk_att else []
        }

    def freeze_all_transformer_layers(self):
        """
        Freeze all layer weight parameters. They will not be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = False

    def unfreeze_all_transformer_layers(self):
        """
        Unfreeze all layers weight parameters. They will be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = True

    def unfreeze_transformer_last_layers(self):
        for name, param in self.transformer_layer.named_parameters():
            if "layer.11" in name or "pooler" in name:
                param.requires_grad = True

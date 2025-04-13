import inspect
import os
from dataclasses import dataclass
import torch
from torch import nn
import math
from torch.nn import  functional as F

@dataclass
class GPTConfig:
    block_size:int = 1024#最大序列长度
    vocab_size:int = 50257#词表长度
    n_layer:int = 12#层数
    n_head:int = 12#头数
    n_embd:int =768#嵌入表示维度
class CausalSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        ##并行对所有头进行q,k,v的投影
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)##3是因为并行对q,k,v进行一个投影
        ##输出进行一个投影
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.MYGPT_SCALE_INIT = 1##残差缩放的标志
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        ##遵循OpenAI的命名准则
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
    def forward(self,x):
        """自注意力"""
        B,T,C = x.size()##batch_size,序列长度,嵌入表示的维度(n_embd)
        qkv = self.c_attn(x)##先计算线性映射后的qkv
        q,k,v = qkv.split(self.n_embd,dim=2)##将q,k,v单独拿出来
        ##nh:头的数量,hs:头的大小,C:hs*ns——通道数量
        ##这里做矩阵的转置是为了之后更快地计算
        ##这里做的一件事就是把nh放到批处理的维度,在之后的操作中,把B,nh视作批处理
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)#(B,nh,T,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)
        ##用缩放点积注意力
        # att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))##自回归掩码,使得在训练时只看的到预测位前面的信息
        # att = F.softmax(att,dim=-1)
        # y = att@v##对values加权求和(B,nh,T,T)x(B,nh,T,hs) -> (B,nh,T,hs)
        ## FlashAttention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)##调用FlashAttention
        y = y.transpose(1,2).contiguous().view(B,T,C)##将张量重新连接起来,contiguous保证内存连续
        ##对输出做一个投影
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    """FFN:GELU(XW_1+b)W_2+b"""
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.MYGPT_SCALE_INIT = 1##残差缩放的标志
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))
class Block(nn.Module):
    """解码器中的每个块"""
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),##嵌入层
            wpe = nn.Embedding(config.block_size,config.n_embd),##位置编码
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),##transformer的主要构成部分
            ln_f = nn.LayerNorm(config.n_embd)##最后的层归一化
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)##输出的全连接层讲嵌入表示映射到单个token上
        ##权重共享
        self.transformer.wte.weight = self.lm_head.weight##这里是复制引用
        # self._orig_mod = nn.ModuleDict(dict(transformer=self.transformer,lm_head=self.lm_head))
        self.apply(self._init_weights)##对所有子模块使用_init_weights函数
    def _init_weights(self,module):
        std = 0.02
        if hasattr(module,'MYGPT_SCALE_INIT'):
            std *= (2*self.config.n_layer)**-0.5##防止残差连接后标准差过大,*2是因为transformer的每一层都有两个残差连接
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)##0均值标准差为0.02,如果使用xavier初始化,由于GPT2中的嵌入表示一般为768或者1600,所以结果差不多
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)##torch中的bias是统一用均匀分布初始化的
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
    def forward(self,idx,targets=None):
        """
        前向传播
        :param idx:输入数据
        :param targets: 输入数据对应的标签
        :return: 输出概率以及loss
        """
        #idx是token的索引,idx的形状是(B,T)=(Batch_size,Time)
        B,T = idx.size()
        assert T <= self.config.block_size,f'Cannot forward sequence of length {T},block size is only {self.config.block_size}'
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)#shape(T)
        ##位置编码和嵌入表示层
        pos_emb = self.transformer.wpe(pos)##位置编码,(T,n_embd)
        token_emb = self.transformer.wte(idx)##嵌入表示,(B,T,n_embd)
        x = token_emb + pos_emb##这里使用了广播机制
        ##主体结构
        for block in self.transformer.h:
            x = block(x)
        ##传到层归一化和softmax层
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)#(B,T,vocab_size)vocab_size表示可能token的数量
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))##交叉熵损失函数只接受二维的张量,所以要先将其展平
        return logits,loss##最后logits取一个Softmax即可变成概率
    @classmethod
    def from_pretrained(cls,model_type):
        """从hugging face上加载模型参数"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weights from model:{model_type}')
        ##按照hugging face上gpt2模型进行配置
        config_args = {
            'gpt2':         dict(n_layer=12,n_head=12,n_embd=768),#124M
            'gpt2-medium':  dict(n_layer=24,n_head=16,n_embd=1024),#350M
            'gpt-large':    dict(n_layer=36,n_head=20,n_embd=1280),#774M
            'gpt-xl':       dict(n_layer=48,n_head=25,n_embd=1600)#1558M
        }[model_type]
        config_args['vocab_size']=50257
        config_args['block_size']=1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()##自己创建模型的状态字典
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]##忽略掉一些缓冲区,attn.bias只用于计算自回归的掩码
        ##初始化hugging face transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        ##从hugging face上复制参数
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        ##因为源代码中使用的是tf,所以权重需要转置一下才能使用
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        ##OpenAI在上面这些权重中用的是Conv1d,但是我们只想用一个全连接层
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #Conv1d的权重需要被转置
                # print(k,sd[k].shape)
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else :
                #其他的就直接复制
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    def configure_optimizers(self,weight_decay,learning_rate,device):
        param_dict = {pn:p for pn,p in self.named_parameters()}##先收集所有参数及其名字
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}##筛选出需要梯度的参数
        ##创建Optim group,所有2维的参数都要使用权重衰减,否则(bias等其他一维张量)不使用权重衰减
        decay_params = [p for _,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _,p in param_dict.items() if p.dim() < 2]
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        # print(sum)
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decayed tensor:{len(decay_params)} with {num_decay_params} decayed')
        print(f'num no-decayed tensor:{len(nodecay_params)} with {num_nodecay_params} no decayed')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW:{use_fused}')
        optimizer = torch.optim.AdamW(optim_groups,lr = learning_rate,betas=(0.9,0.95),eps=1e-8)
        return optimizer
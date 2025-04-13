
import os
import torch

import math
from torch.nn import  functional as F

from hellaswag import iterate_examples, render_example
from GPT_2 import GPT,GPTConfig
import tiktoken
num_return_sequence = 5
max_length = 30
##检测一下设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device:{device}')

# model = GPT.from_pretrained('gpt2')
##使用自己训练的GPT
##在GPT2中压缩率为3:1,1000个字符被映射为300个token
##添加种子
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
import tiktoken
import time
import numpy as np
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt
class DataLoaderLite:
    def __init__(self,B,T,split):
        self.current_position = None
        self.tokens = None
        self.current_shard = None
        self.B = B
        self.T = T
        assert split in {'train','val'}
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards)>0,f'no shards found for split {split}'
        self.reset()
    def reset(self):
        """重置状态"""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)##输入
        y = (buf[1:]).view(B,T)##label
        self.current_position += B*T
        ##如果读完了就回到第一个
        if self.current_position+(B*T+1)>len(self.tokens):
            self.current_shard  = (self.current_shard+1)%len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x,y
def get_most_likely_row(tokens,mask,logits):
    shift_logits = (logits[...,:-1,:]).contiguous()
    shift_tokens = (tokens[...,:1,:]).contiguous()
    flat_shift_logits = shift_logits.view(-1,shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits,flat_shift_tokens,reduction='none')
    shift_losses = shift_losses.view(tokens.size(0),-1)
    shift_mask = (mask[...,1:]).contiguous()
    masked_shift_losses = shift_losses*shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss/shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

enc = tiktoken.get_encoding('gpt2')
# with open('datas/input.txt','r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B,T=4,32##调试用
# buf = torch.tensor(tokens[:B*T+1],device=device)##记得将他转到device上
# x = buf[:-1].reshape((B,T))##用作数据
# y = buf[1:].reshape((B,T))##用作label
##按照GPT-3的配置,我们要加载0.5M个Token
total_batch_size = 54288##2**19
##不能一次加载所有0.5M token,要处理多个序列,让它们的梯度相加以模拟0.5M的批处理大小
B,T = 16,1024##B为微批次大小,T为序列长度,我们有BxT个Tokens进入Transformer进行前向传播和后向传播,但是不会更新
grad_accu_steps = total_batch_size//(B*T)

train_loader = DataLoaderLite(B=B,T=T,split='train')
valid_loader = DataLoaderLite(B=B,T=T,split='val')
# ##启用TF32
torch.set_float32_matmul_precision('high')##highest使用FP32,high使用TF32
#
model = GPT(GPTConfig(vocab_size=50304))##变成偶数便于计算,多创建的内存永远不会被访问,受影响的只有解码器的输入时的嵌入层,和输出的全连接层,由于这些多出来的内存概率都为0,所以不会造成影响

model.to(device)
raw_model = model
model = torch.compile(model)
# logits ,loss= model(x,y)
max_steps = 19073
##根据GPT-3中参数设置
max_lr = 6e-4
min_lr = max_lr*0.1
warmup_steps = 715
def get_lr(it):
    ##预热阶段线性预热
    if it < warmup_steps:
        return max_lr*(it+1)/warmup_steps
    ##优化完后的lr
    if it > max_steps:
        return min_lr
    ##预热阶段后到优化完时,使用余弦衰减
    decay_ratio = (it-warmup_steps)/(max_steps-warmup_steps)
    assert  0 <= decay_ratio <= 1
    coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))##衰减因子从1到0
    return min_lr + coeff*(max_lr-min_lr)
#optimize
# optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)##AdamW是对Adam优化器的一个优化
optimizer = model.configure_optimizers(weight_decay=0.01,learning_rate=6e-4,device=device)
##创建日志
log_dir = "log"
os.makedirs(log_dir,exist_ok=True)
log_file = os.path.join(log_dir,f"log.txt")
with open(log_file,'w') as f:##以写的方式打开是为了清除掉文件内容
    pass
for step in range(max_steps):
    t0 = time.time()
    last_step = (step==max_steps-1)
    ##评估
    if step % 250 == 0 or last_step:
        model.eval()
        valid_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x,y = valid_loader.next_batch()
                x,y = x.to(device),y.to(device)
                logits,loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f'validation loss : {val_loss_accum.item():.4f}')
        with open(log_file,'w') as f:
            f.write(f'validation loss : {val_loss_accum.item():.4f}\n')
    if step > 0 and (step%500==0 or last_step):
        ##将模型保存
        check_point_path = os.path.join(log_dir,f'model_{step:05d}.pt')
        check_point = {
            'model':raw_model.state_dict(),
            'config':raw_model.config,
            'step':step,
            'val_loss':val_loss_accum.item()
        }
        torch.save(check_point,check_point_path)
    ##hellaswag评估
    if step % 250 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0
        for i,example in enumerate(iterate_examples('val')):
            _,tokens,mask,label = render_example(example)
            mask = mask.to(device)
            tokens = tokens.to(device=device)
            with torch.no_grad():
                logits,loss = model(tokens)
                pred_norm = get_most_likely_row(tokens,mask,logits)
            num_total += 1
            num_correct_norm += int(pred_norm==label)
            acc_norm = num_correct_norm/num_total
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    ##生成数据
    if step > 0 and step%100 == 0 :
        model.eval()
        num_return_sequence = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens,dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            ##计算结果
            with torch.no_grad():
                logits,loss = model(xgen)##(B,T,vocab_size)
                ##取当前时间
                logits = logits[:,-1,:]##(B,vocab_size)
                ##产生概率
                probs = F.softmax(logits,dim=-1)
                ##选top50
                ##topk_probs和topk_indices都是(5,50)
                topk_probs , topk_indices = torch.topk(probs,50,dim=-1)
                ix = torch.multinomial(topk_probs, 1,generator=sample_rng)##(B,T)
                ##将值放到正确的索引上
                xcol = torch.gather(topk_indices,-1,ix)##在最后一维按照ix放置topk_indices中的数据
                ##最后将它连在后面即可
                xgen = torch.cat((xgen,xcol),dim=1)##最后得到的x是5x30的
        for i in range(num_return_sequence):
            tokens = xgen[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f'sample {i} :',decoded)

    ##训练
    model.train()
    optimizer.zero_grad()##从0梯度开始
    loss_accum = 0.0
    for micro_steps in range(grad_accu_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)##在训练时把数据加载到GPU上防止浪费太多的GPU内存
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss = model(x,y)##在计算前向传播和Loss的时候用bfloat16
        loss = loss/grad_accu_steps##防止多个微批次处理一个批次时,梯度没有归一化
        loss_accum += loss.detach()
        loss.backward()##每次反传都会在参数的梯度上加一个数
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)##梯度裁剪防止大梯度影响模型训练
    lr = get_lr(step)
    ##更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr##优化器中有多个参数组,要将每个参数组的lr都更改
    optimizer.step()##更新参数
    torch.cuda.synchronize()##等待GPU完成上面的代码
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = (train_loader.B*train_loader.T)/(t1-t0)
    print(f'step{step} | loss:{loss_accum.item()} | lr : {lr:.6f} | norm : {norm:.4f} | dt : {dt:.2f}ms | tok/sec : {tokens_per_sec}')##loss.item()将其转为float
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")
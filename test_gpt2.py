import torch
from torch.nn import functional as F
import tiktoken
from GPT_2 import GPT,GPTConfig
##预测
model = GPT(GPTConfig(vocab_size=50304))
num_return_sequence = 5
max_length = 30
##检测一下设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device:{device}')
enc = tiktoken.get_encoding('gpt2')
weight = torch.load('weight/model_weights.pth',map_location=torch.device('cpu'))
new_weight = {k.replace('_orig_mod.',''):v for k,v in weight.items()}
model.load_state_dict(new_weight)
model.eval()
tokens = enc.encode(input("input token"))##将句子编码成整数
tokens = torch.tensor(tokens,dtype=torch.long)#shape(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence,1)#将其复制5层得到形状(5,8)
##将其作为初始输入
x = tokens.to(device)
##x大小为(B,T)也就是(5,8)
##在每次循环中将一列新的索引添加到这些行中,每次将新采样到的索引加到x中
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)<max_length:
    with torch.no_grad():##告诉torch不会发生backward,所以不用缓存这些张量
        logits,loss = model(x)#(B,T,vocab_size)
        ##只关心最新的logits也即T第二维的最后一个
        logits = logits[:,-1,:]#(B,vocab_size)
        probs = F.softmax(logits,dim=-1)
        ##选概率最高的50个,因为这时hugging face标准
        topk_probs,topk_indices = torch.topk(probs,50,dim=-1)##只保留最高的50个概率,低于的全部置为0,并重新规范化
        ##从top-k中选择一个Token
        ix = torch.multinomial(topk_probs,1)
        ##将值放到正确的索引上
        xcol = torch.gather(topk_indices,-1,ix)##在最后一维按照ix放置topk_indices中的数据
        ##最后将它连在后面即可
        x = torch.cat((x,xcol),dim=1)##最后得到的x是5x30的
for i in range(num_return_sequence):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print('>',decoded)
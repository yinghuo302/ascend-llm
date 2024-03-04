import torch
from torch import nn,Tensor
from typing import Optional,List,Tuple

def quantize(mat:Tensor)-> Tuple[Tensor,Tensor]:
    max_val = torch.max(torch.abs(mat),dim=-1)[0]
    # mat =  (mat * (127 / max_val)[...,None]).to(dtype=torch.int8)
    mat =  torch.round(mat * (127 / max_val)[...,None]) # 该代码用于GPU测试
    return mat, max_val

def dequantize(mat:Tensor,max_val:Tensor):
    return torch.mul(mat,max_val.unsqueeze(-1))/ 127.0

def decomposition_w_topk(mat:Tensor,k:int = 16) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    index=torch.topk(torch.max(torch.abs(mat),dim=-2)[0],k=64)[1]
    unq_idx=torch.zeros(mat.shape[-1],dtype=torch.bool)
    unq_idx[index]=True
    q_idx=~unq_idx
    return mat,mat[...,unq_idx],q_idx,unq_idx


# 会将mat中的内容修改
def decomposition_topk(mat:Tensor,k:int=16) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    mat_v=mat.view(-1,mat.shape[-1])
    idx_unq=torch.topk(torch.max(torch.abs(mat_v),dim=-2)[0],k,dim=-1)[1]
    mat_unq = mat[...,idx_unq]
    mat[...,idx_unq]=0
    return mat,mat_unq,idx_unq

def decomposition_thres(mat:Tensor,threshold:float=6.0) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    mat_v=mat.view(-1,mat.shape[-1])
    idx_unq = torch.max(torch.abs(mat_v),dim=-2)[0] >= threshold
    mat_unq = mat[...,idx_unq]
    mat[...,idx_unq]=0
    return mat,mat_unq,idx_unq


class MatMulInteger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor,weight:torch.Tensor):               
        return torch.matmul(x.to(device="cpu",dtype=torch.int32) , # 输出为torch.int32，torch不支持CUDA上的int8矩阵乘
                                    weight.to(device="cpu",dtype=torch.int32))

    @staticmethod
    def symbolic(g:torch._C.Graph, x:torch.Tensor,weight:torch.Tensor):
        return g.op("MatMulInteger", x,weight)

matmulInteger = MatMulInteger.apply

# F.linear for x_q,w_q,bias
def qlinear(x_q:Tensor,x_max:Tensor,weight_q:Tensor,w_max:Tensor,dtype,bias=None):
    # res_q = matmulInteger(x_q , weight_q.t())
    res_q=nn.functional.linear(x_q.to(torch.float32),weight_q.to(torch.float32))
    # res_q=nn.functional.linear(x_q,weight_q)
    mx = nn.functional.linear(x_max.unsqueeze(-1),w_max.unsqueeze(-1))
    res = torch.mul((res_q.to(device=mx.device,dtype=torch.float32) / (127.0*127.0)).to(torch.float16), mx )  
    if bias is not None:
        res = res + bias
    return res

class W8Linear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,theshold = 6.0) -> None:
        super().__init__()
        self.bias = None if not bias else bias.detach()
        self.dtype = origin_weight.dtype
        self.theshold = theshold
        self.weight_q,self.max_val = quantize(origin_weight)
        self.weight_q = nn.Parameter(self.weight_q,requires_grad=False).detach()
        self.max_val = nn.Parameter(self.max_val,requires_grad=False).detach()

    def forward(self,x:Tensor) -> Tensor:
        return nn.functional.linear(x,dequantize(self.weight_q,self.max_val),bias=self.bias)

class W8X8Linear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,theshold = 6.0) -> None:
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.theshold = theshold
        self.weight_q,self.max_val = quantize(origin_weight.detach())
        self.weight_q = nn.Parameter(self.weight_q,requires_grad=False).detach()
        self.max_val = nn.Parameter(self.max_val,requires_grad=False).detach()

    def forward(self,x:Tensor) -> Tensor:
        x_q,x_max = quantize(x)
        return qlinear(x_q,x_max,self.weight_q,self.max_val,x.dtype,self.bias)


class W8DXLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,theshold = 6.0) -> None:
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.theshold = theshold
        # self.origin_weight = 
        self.weight_q,self.max_val = quantize(origin_weight.detach())
        self.weight_q = nn.Parameter(self.weight_q,requires_grad=False).detach()
        self.max_val = nn.Parameter(self.max_val,requires_grad=False).detach()

    def forward(self,x:Tensor) -> Tensor:
        x1=x.clone()
        x_q,x_unq,idx_unq = decomposition_topk(x1)
        x_q,x_max = quantize(x_q)
        res_q = qlinear(x_q,x_max,self.weight_q,self.max_val,x.dtype)
        weight_unq=dequantize(self.weight_q[...,idx_unq],self.max_val)
        res_unq = nn.functional.linear(x_unq, weight_unq,self.bias)
        # origin_res = nn.functional.linear(origin_x,self.origin_weight,self.bias)
        # print(torch.div(res + res_unq -origin_res,origin_res))
        return res_q + res_unq


class W8DWLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,theshold = 6.0) -> None:
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.theshold = theshold
        # self.origin_weight = origin_weight.detach()
        # self.weight_q,self.weight_unq,self.idx_q,self.idx_unq =decomposition_w_topk(origin_weight.detach())
        self.weight_q,self.weight_unq,self.idx_unq = decomposition_topk(origin_weight.detach())
        self.weight_q,self.w_max = quantize(self.weight_q)
        self.weight_q = nn.Parameter(self.weight_q,requires_grad=False).detach()
        self.weight_unq = nn.Parameter(self.weight_unq,requires_grad=False).detach()
        self.w_max = nn.Parameter(self.w_max,requires_grad=False).detach()

    def forward(self,x:Tensor) -> Tensor:
        x1=x.clone()
        x_unq = x1[...,self.idx_unq]
        x1[...,self.idx_unq]=0
        x_q,x_max = quantize(x1)
        res_q = qlinear(x_q,x_max,self.weight_q,self.w_max,x1.dtype)
        res_unq = nn.functional.linear(x_unq, self.weight_unq,self.bias)
        return res_q + res_unq

def replace_linear_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name,W8DXLinear(child.weight,child.bias,theshold=6.0))
        else:
            replace_linear_modules(child)

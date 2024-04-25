import torch
from torch import nn,Tensor
from typing import Optional,List,Tuple
from torch.onnx.symbolic_helper import parse_args

class MatMulInteger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor,weight_t:torch.Tensor):               
        res = torch.matmul(x.to(dtype=torch.float32),weight_t.to(torch.float32))
        # res=torch.matmul(x.to(dtype=torch.int32,device="cpu") , # torch不支持CUDA上的int8矩阵乘
        #                             weight_t.to(dtype=torch.int32,device="cpu")).to(x.device)
        return res

    @staticmethod
    @parse_args("v","v")
    def symbolic(g:torch._C.Graph, x:torch.Tensor,weight_t:torch.Tensor):
        return g.op("MatMulInteger", x,weight_t)

matmulInteger = MatMulInteger.apply

def quantize_mat(mat:Tensor)-> Tuple[Tensor,Tensor]:
    # max_val = torch.max(torch.abs(mat),dim=-1)[0]
    # mat =  (mat * (127 / max_val)[...,None]).to(dtype=torch.int8)
    max_val = (torch.max(torch.abs(mat),dim=-1)[0] / 127.0).to(dtype=torch.float16)
    mat =  (mat / max_val[...,None]).to(dtype=torch.int8)
    return mat, max_val

def dequantize_mat(mat:Tensor,max_val:Tensor):
    return torch.mul(mat,max_val.unsqueeze(-1))

def decomposition(mat:Tensor,unq_idx:Tensor,t:Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    return mat.mul(t.to(dtype=mat.dtype)),mat[...,unq_idx]
    mat=mat.clone()
    mat_unq = mat[...,unq_idx]
    if mat.dim() == 3:
        mat[:,:,unq_idx] = 0
    elif mat.dim() == 4:
        mat[:,:,:,unq_idx] = 0
    elif mat.dim() == 2:
        mat[:,unq_idx] = 0
    return mat,mat_unq

def get_unq_idx_topk(mat:Tensor,k:int=64):
    idx=torch.topk(mat.view(-1,mat.shape[-1]).abs().max(dim=-2)[0],k,dim=-1)[1]
    t = torch.ones((mat.shape[-1]),dtype=mat.dtype,device=mat.device)
    t = t.clone()
    t[idx] = 0
    return idx,t

def get_unq_idx_thres(mat:Tensor,threshold:float=6.0):
    k = mat.view(-1,mat.shape[-1]).abs().max(dim=-2)[0] >= threshold
    return k.nonzero().view(-1), k

def qMatmul(x_q:Tensor,x_max:Tensor,weight_q:Tensor,w_max:Tensor,dtype):
    res_q = matmulInteger(x_q , weight_q)
    mx = nn.functional.linear(x_max.unsqueeze(-1),w_max.unsqueeze(-1))
    res = torch.mul(res_q.to(device=mx.device,dtype=torch.float32), mx.to(torch.float32) ).to(dtype=dtype)  
    # res = torch.mul((res_q.to(device=mx.device,dtype=torch.float32) / (127.0*127.0)).to(torch.float16), mx )  
    return res

class W8Linear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        self.weight_q,self.max_val = quantize_mat(origin_weight.detach())
        self.weight_q = nn.Parameter(self.weight_q,requires_grad=False)
        self.max_val = nn.Parameter(self.max_val,requires_grad=False)

    def forward(self,x:Tensor) -> Tensor:
        return nn.functional.linear(x,dequantize_mat(self.weight_q,self.max_val),bias=self.bias)

# act_max for smooth 
class W8X8Linear(nn.Module):
    def __init__(self, ori_w:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = ori_w.dtype
        self.alpha = alpha
        self.scales = None
        if act_max is not None:
            act_max = act_max.to(ori_w.device)
            self.scales = (act_max.pow(alpha) / ori_w.abs().max(dim=0)[0].pow(1 - alpha)).clamp(min=1e-5).to(dtype=ori_w.dtype)
            self.scales = nn.Parameter(self.scales,requires_grad=False).detach()
            ori_w = ori_w.detach().mul(self.scales)
        self.weight_q,self.max_val = quantize_mat(ori_w.detach())
        self.weight_q = nn.Parameter(self.weight_q.t(),requires_grad=False).detach()
        self.max_val = nn.Parameter(self.max_val,requires_grad=False).detach()

    def forward(self,x:Tensor) -> Tensor:
        if self.scales is not None:
            x = x.div(self.scales)
        x_q,x_max = quantize_mat(x)
        res = qMatmul(x_q,x_max,self.weight_q,self.max_val,x.dtype)
        if self.bias is not None:
            res = res + self.bias
        return res

# static decomposition
class W8SDLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        if act_max is not None:
            self.idx_unq,self.t = get_unq_idx_topk(act_max,self.alpha)
        else:
            self.idx_unq,self.t = get_unq_idx_topk(origin_weight,self.alpha)
        self.idx_unq,self.t = self.idx_unq.to(origin_weight.device),self.t.to(origin_weight.device)
        self.weight_q,self.weight_unq = decomposition(origin_weight,self.idx_unq,self.t)
        self.weight_q,self.w_max = quantize_mat(self.weight_q.detach())
        self.weight_q = nn.Parameter(self.weight_q.t(),requires_grad=False)
        self.weight_unq = nn.Parameter(self.weight_unq.t(),requires_grad=False)
        self.w_max = nn.Parameter(self.w_max,requires_grad=False)
        self.t = nn.Parameter(self.t,requires_grad=False)
        self.idx_unq = nn.Parameter(self.idx_unq,requires_grad=False)

    def forward(self,x:Tensor) -> Tensor:
        x_q,x_unq = decomposition(x,self.idx_unq,self.t)
        x_q,x_max = quantize_mat(x_q)
        res_q = qMatmul(x_q,x_max,self.weight_q,self.w_max,x.dtype)
        res_unq = torch.matmul(x_unq, self.weight_unq)
        if self.bias is not None:
            res_unq += self.bias
        return res_q + res_unq
    
class W8DXLinear(nn.Module):
    def __init__(self, origin_weight:Tensor, bias: Optional[Tensor] = None,act_max:Optional[Tensor] = None,alpha=32):
        super().__init__()
        self.bias = None if bias is None else bias.detach()
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        self.weight_q,self.max_val = quantize_mat(origin_weight.detach())
        self.weight_q = nn.Parameter(self.weight_q.t(),requires_grad=False)
        self.max_val = nn.Parameter(self.max_val,requires_grad=False)

    def forward(self,x:Tensor) -> Tensor:
        idx_unq,t = get_unq_idx_topk(x,self.alpha)
        x_q,x_unq = decomposition(x,idx_unq,t)
        x_q,x_max = quantize_mat(x_q)
        res_q = qMatmul(x_q,x_max,self.weight_q,self.max_val,x.dtype)
        weight_unq= torch.mul(self.weight_q[idx_unq,:],self.max_val.unsqueeze(0))
        res_unq = torch.matmul(x_unq, weight_unq)
        if self.bias is not None:
            res_unq += self.bias
        return res_q + res_unq


quant_cls = {
    "W8":W8Linear,
    "W8X8":W8X8Linear,
    "W8SD":W8SDLinear,
    "W8DX":W8DXLinear
}

# mp = {"q_proj":4.,"k_proj":4.,"v_proj":4.,"o_proj":4.,"gate_proj":4.,"up_proj":4.,"down_proj":4.}
def replace_linear_modules(module:nn.Module,prefix:str,act_scales,cfg):
    for name, child in module.named_children():
        prefix_next = (prefix + '.' + name) if prefix != '' else name
        if isinstance(child, nn.Linear) and name in cfg:
            act_scale = None if act_scales is None or 'act_scale' not in cfg[name] else act_scales[prefix_next]
            alpha = 128 if 'alpha' not in cfg[name] else cfg[name]['alpha']
            # mp[name] = mp[name]-0.125
            # alpha = int(alpha >> int(min(mp[name],3.)))
            setattr(module, name,quant_cls[cfg[name]['type']]
                    (child.weight,child.bias,act_max=act_scale,alpha=alpha))
        else:
            replace_linear_modules(child,prefix_next,act_scales,cfg)

def quantize(model:nn.Module,smooth:bool=False,act_scales_path:Optional[str]=None,cfg={}):
    act_scales = None
    if 'act_scales_path' in cfg:
        act_scales = torch.load(cfg['act_scales_path'])
        if 'smooth' in cfg:
            from smooth import smooth_lm
            smooth_lm(model, act_scales, 0.85)
    replace_linear_modules(model,'',act_scales,cfg)
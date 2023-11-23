import torch
import torch.nn as nn
import torch.nn.functional as F

class RepAdapterLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super(RepAdapterLinear, self).__init__(in_features=in_features, out_features=out_features,)
        self.adapter = RepAdapter(in_features=in_features)
    
    def forward(self, input):
        input = self.adapter(input)
        return F.linear(input, self.weight, self.bias)

    @staticmethod
    def from_linear(linear_module):
        new_linear = RepAdapterLinear(linear_module.in_features, linear_module.out_features)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear
    
    @staticmethod
    def to_linear(repadapter_module):
        repadapter_module.set_RepWeight()
        new_linear = nn.Linear(repadapter_module.in_features, repadapter_module.out_features)
        new_linear.weight = repadapter_module.weight
        new_linear.bias = repadapter_module.bias
        return new_linear
    
    def set_RepWeight(self):
        def sparse2dense(weight,groups):
            d,cg=weight.shape
            dg=d//groups
            weight=weight.view(groups,dg,cg).transpose(1,2)
            new_weight=torch.zeros(cg*groups,d,device=weight.device,dtype=weight.dtype)
            for i in range(groups):
                new_weight[i*cg:(i+1)*cg,i*dg:(i+1)*dg]=weight[i]
            return new_weight.T
        wa = self.adapter.conv_A.weight.squeeze()
        wb = self.adapter.conv_B.weight.squeeze() if self.adapter.conv_B.groups<=1 else sparse2dense(self.adapter.conv_B.weight.squeeze(), self.adapter.conv_B.groups)
        weight,bias = self.reparameterize(wa.T, wb.T,
                        self.adapter.conv_A.bias, self.adapter.conv_B.bias, self.adapter.scale, do_residual=True)
        weight,bias = self.reparameterize(weight.T,self.weight.T,
                                                bias,self.bias)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.xavier_uniform_(self.adapter.conv_A.weight)
        nn.init.zeros_(self.adapter.conv_A.bias)
        nn.init.zeros_(self.adapter.conv_B.weight)
        nn.init.zeros_(self.adapter.conv_B.bias)


    @staticmethod
    def reparameterize(Wa,Wb,Ba,Bb,scale=1,do_residual=False):
        bias = 0
        id_tensor=0
        if Ba is not None:
            bias=Ba@Wb
        if Bb is not None:
            bias=bias+Bb
        if do_residual:
            id_tensor=torch.eye(Wa.shape[0],Wb.shape[1]).to(Wa.device)
        weight = Wa @ Wb*scale + id_tensor
        return weight.T,bias*scale if isinstance(bias,torch.Tensor) else None


class RepadpterModuleInjection:
    @staticmethod
    def make_scalable(linear_module):
        """Make a (linear) layer adding repadapter.
        :param linear_module: A Linear module
        :return: a repadapter linear layer
        """
        new_linear = RepAdapterLinear.from_linear(linear_module)
        return new_linear
    
class RepadpterReparam:
    @staticmethod
    def reparam(repadapter_module):
        """reparam a repadapter layer to a linear layer.
        :param repadapter_module: A repadapter module 
        :return: a linear layer
        """
        new_linear = RepAdapterLinear.to_linear(repadapter_module)
        return new_linear
    
class RepAdapter_plus(nn.Module):
    """
    Pytorch Implemention of RepAdapter for 1d tensor
    copy from https://github.com/luogen1996/RepAdapter/blob/main/repadapter.py
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1,
            temperature=5.
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.conv_C = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.routing_weights=nn.Parameter(torch.rand(2))

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.temperature=temperature
        self.scale_B=scale
        # adaption with different scales
        self.scale_C=float(scale)/2.


        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
        nn.init.zeros_(self.conv_C.weight)
        nn.init.zeros_(self.conv_C.bias)

    def forward(self, x):
        weights=torch.softmax(self.routing_weights/self.temperature,-1)
        x=x.transpose(1,2)
        x_=self.conv_A(x)
        x=self.conv_B(self.dropout(x_))*self.scale_B*weights[0]+self.conv_C(self.dropout(x_))*self.scale_C*weights[1]+x
        x=x.transpose(1,2).contiguous()
        return x

class RepAdapter(nn.Module):
    """
    Pytorch Implemention of RepAdapter for 1d tensor
    copy from https://github.com/luogen1996/RepAdapter/blob/main/repadapter.py
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x,weights=None):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))*self.scale+x
        x=x.transpose(1,2).contiguous()
        return x
    

def set_repadapter(model):
    layers = []
    set_param = 0
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if not 'head' in last_token:
            setattr(parent_layer, last_token, RepadpterModuleInjection.make_scalable(getattr(parent_layer, last_token)))
            set_param +=1
    print(f'successfully set {set_param} layers params')


@torch.no_grad()
def save_repadapter(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['adapter']]):
            trainable[n] = p.data
    torch.save(trainable, save_path)
    
def load_repadapter(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['adapter']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model

def merge_repadapter(model,load_path=None,has_loaded=False):
    # 仅当还没有加载状态且提供了加载路径时，才执行加载操作
    if not has_loaded and load_path is not None:
        set_repadapter(model)
        load_repadapter(load_path,model)
    reparam_num=0
    for name, l in model.named_modules():
        if isinstance(l, torch.nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            parent_layer = layer
            repadapter_layer =  getattr(parent_layer, tokens[-1]) 
            if hasattr(repadapter_layer,"adapter"):
                last_token = tokens[-1]
                setattr(parent_layer, last_token,  RepAdapterLinear.to_linear(repadapter_layer))
                reparam_num = reparam_num + 1
    print(f'successfully reparam {reparam_num} layers')


if __name__ == '__main__':
    import transformers
    input_tensor = torch.randint(1, 10, (1,20))
    # repadpter_linear = RepAdapterLinear(20, 10)
    # nn.init.normal_(repadpter_linear.adapter.conv_A.weight)
    # nn.init.normal_(repadpter_linear.adapter.conv_B.weight)
    # repadpter_linear.eval()
    # output_repadpter1 = repadpter_linear(input_tensor)
    # print("RepAdapterLinear 输出结果1:", output_repadpter1)
    # linear_module = RepAdapterLinear.to_linear(repadpter_linear)
    # output_repadpter2 = repadpter_linear(input_tensor)
    # print("RepAdapterLinear 输出结果2:", output_repadpter2)
    # output_linear = linear_module(input_tensor)
    # print("Linear            输出结果:", output_linear)
    model_name_or_path = "/mnt/SFT_store/Linksoul-llama2-7b"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,)
    model(input_tensor)
    # print([name for name, l in model.named_modules()])
    # load_path = "/mnt/SFT_store/flageval_peft/outputs/repadapter/2023-10-19_02-45-41_success/final.pt"
    # merge_repadapter(model,load_path)
    # print([name for name, l in model.named_modules()])
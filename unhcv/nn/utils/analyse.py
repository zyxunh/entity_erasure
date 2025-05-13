from collections import OrderedDict

import torch
# from unhcv.models.nn.utils.analyse import write_model_structure


def print_memory_status():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

def freeze_model(model):
    """
    Freeze the model
    """
    # model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def filter_param(parameters, mode="requires_grad"):
    if mode == "requires_grad":
        return filter(lambda p: p.requires_grad, parameters)
    elif mode == "not_requires_grad":
        return filter(lambda p: not p.requires_grad, parameters)
    else:
        raise ValueError("mode should be requires_grad or not_requires_grad")


def analyse_optim_param(optim, every=False):
    grad_lt = []
    param_shape_lt = []
    for group in optim.param_groups:
        grad_lt.append(dict(sum=0, grad_num=0, values=[], without_grad_num=0))
        param_shape_lt.append(dict(sum=0, values=[], grad=0, without_grad=0))
        for p in group['params']:
            param_shape_lt[-1]["values"].append(p.shape)
            param_shape_lt[-1]["sum"] += p.numel()
            if every:
                grad_lt[-1]["values"].append(p.grad)
            if p.grad is not None:
                grad_lt[-1]["sum"] += (p.grad.data ** 2).sum()
                grad_lt[-1]["grad_num"] += 1
                param_shape_lt[-1]["grad"] += p.numel()
            else:
                grad_lt[-1]["without_grad_num"] += 1
                param_shape_lt[-1]["without_grad"] += p.numel()
    return grad_lt, param_shape_lt

def analyse_model_param(model):
    grad_dict = OrderedDict(sum=0, values=OrderedDict(), true_num=0, false_num=0)
    param_inform_dict = dict(requires_grad=OrderedDict(sum=0, values=OrderedDict(), num=0),
                             not_requires_grad=OrderedDict(sum=0, values=OrderedDict(), num=0))
    for name, p in model.named_parameters():
        if p.requires_grad:
            container = param_inform_dict["requires_grad"]
        else:
            container = param_inform_dict["not_requires_grad"]
        container["values"][name] = p.shape
        container["sum"] += p.numel()
        container["num"] += 1
        if p.requires_grad:
            if p.grad is not None:
                grad_dict["values"][name] = (p.grad.data ** 2).sum()
                grad_dict["sum"] += grad_dict["values"][name]
                grad_dict["true_num"] += 1
            else:
                grad_dict["values"][name] = None
                grad_dict["false_num"] += 1
    return grad_dict, param_inform_dict

def cal_para_num(model):
    requires_num = 0
    not_requires_num = 0
    for name, para in model.named_parameters():
        if para.requires_grad:
            requires_num += para.numel()
        else:
            not_requires_num += para.numel()
    return 'requires_grad: {} M, not_requires_grad: {} M'.format(requires_num / 1e6, not_requires_num / 1e6)

def sum_para_value(model):
    sum_value = [0, 0]
    for name, para in model.named_parameters():
        if para.requires_grad:
            sum_value[0] += para.sum()
        else:
            sum_value[1] += para.sum()
    return 'requires_grad: {}, not_requires_grad: {}'.format(sum_value[0], sum_value[1])

def sum_para_grad_value(model):
    sum_value = 0
    for name, para in model.named_parameters():
        if para.requires_grad and para.grad is not None:
            sum_value += para.grad.sum()

    return 'requires_grad\'s grad value: {}'.format(sum_value)

def sum_para_num(paras):
    num = 0 
    for var in paras:
        num += var.numel()
    return num

def get_para(model):
    para_requires_grad = {}
    para_not_requires_grad = {}
    for name, para in model.named_parameters():
        if para.requires_grad:
            para_requires_grad[name] = para
        else:
            para_not_requires_grad[name] = para
    return para_requires_grad, para_not_requires_grad

def write_model_structure(model: torch.nn.Module, file_path):
    with open(file_path, 'w') as f:
        f.write(model.__str__())

if __name__ == '__main__':
    model = torch.nn.Conv2d(1, 1, 1)
    write_model_structure(model, '/home/tiger/debug.txt')


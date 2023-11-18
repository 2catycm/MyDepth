import torch
import torch.nn as nn
class MMoE(nn.Module):
    def __init__(self, num_experts, num_tasks, expert_models, gate_models, tower_models):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        # 初始化Expert模型列表
        self.experts = nn.ModuleList([expert_model() for expert_model in expert_models])
        # 初始化Gate模型列表
        self.gates = nn.ModuleList([gate_model() for gate_model in gate_models])
        # 初始化Task-Specific模型列表
        self.towers = nn.ModuleList([tower_model() for tower_model in tower_models])

    def forward(self, x):
        batch, in_channel, height, width = x.shape
        # Expert模型的输出列表
        # https://blog.csdn.net/feng_xun123/article/details/108960632 pytorch采用NCHW
        expert_outputs = [expert(x) for expert in self.experts] # [<batch, 1, height, width>]
        # Gate模型的输出列表，用于权重计算
        gate_outputs = [gate(x) for gate in self.gates] # [<batch, num_experts>]
        
        # 将Expert模型的输出组合为单个torch张量
        expert_outputs_combined = torch.stack(expert_outputs, dim=-1) # [batch, 1, height, width, num_experts]

        # 将Gate模型的输出组合为单个torch张量
        gate_outputs_combined = torch.stack(gate_outputs, dim=-1) # [batch, num_experts, num_tasks]
        
        # 调整Expert模型输出的形状
        # https://pytorch.org/docs/stable/generated/torch.bmm.html 
        # 因为bmm只支持三维，所以必须暂时把视觉特征搞成一列
        expert_outputs_reshaped = expert_outputs_combined.view(
            batch, -1, self.num_experts) # [batch, 1*height*width, num_experts]

        # 对于batch中的每一个元素独立进行矩阵乘法操作，来通过gating进行线性组合
        combined_features = torch.bmm(expert_outputs_reshaped, gate_outputs_combined) # [batch, 1*height*width, num_tasks]

        # 通过Task-Specific模型列表得到最终任务输出列表
        # task_outputs = [tower(combined_features[..., task_num].reshape()) 
        task_outputs = [tower(combined_features[:, :, task_num].view(batch, 1, height, width)) 
                        for task_num, tower in enumerate(self.towers)] # [<batch, 1, height, width>]

        return task_outputs


# # TODO 

    
# class MultiDepth(nn.Module):
#     def __init__(self, pretrained_weights_path, 
#                 #  head_class=MyNetwork_large(), gating
#                  ):
#         super().__init__()
#         self.omni = get_omni(pretrained_weights_path)
#         set_require_grad(self.omni, False)
#         self.head = head
#         set_require_grad(self.head, True)
        
#     def forward(self, x):
        

#         return dict(
#             # metric_depth=output.squeeze(dim=1)
#             metric_depth=absolute_depth_map,
#             rel_depth=relative_depth_map
#         )
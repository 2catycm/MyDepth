from boilerplate import *
# 定义训练参数，方便调参
# lr = 3e-6
# lr = 1e-5 # OmniScale用
# lr = 3e-4 
# lr = 1e-5 # ThreeDPT 1%大loss达到0.46。原本应当是0.11
# lr = 1e-3 # ThreeDPT 1%大loss达到0.498。原本应当是0.11 第一步是0.4
# lr = 1e-2
# lr = 1e-1 # ZoeDepthOmni可用 0.48
# DPT:1% 0.35
# lr = 0.5
# lr = 0.9
lr = 0.0002512 # ZoeDepth设置, DPT也可以用
# SGD %3=>0.46, 每次迭代降低0.001
# AdamW %13=>0.35, 卡住了，
batch_size*=24564/13042 # Zoe升
batch_size = int(batch_size)


# lr = lr * batch_size/8
lr = lr * batch_size/16
print(f"learning_rate={lr}")
num_epochs = 2
# num_epochs = 4
# %%
# 定义数据路径
# exp_id = "复现实验"
# exp_id = "不使用PEFT-不使用lr_schedule-仅训练两轮"
# exp_id = "复现初赛-添加sam"
# exp_id = "最激进"
# exp_id = "最激进-Zoe"
# exp_id = "最激进-OmniScale"
# exp_id = "最激进-DPT-鱼眼优化"
# exp_id = "3DPT稳定版-根据鱼眼做进一步微调"
exp_id = "激进Zoe-根据鱼眼做进一步微调"
model_name = "ZoeDepth_Omni"
# model_name = "ThreeDPT"
# model_name = "ThreeDPT"
# model_name = "OmniScale"

# running_path = this_directory/f"./runs/{exp_id}"  # 运行时保存的位置
running_path = system_data_path / f"./runs/{exp_id}"  # 运行时保存的位置
if not running_path.is_symlink():  # 如果是软连接，也是存在
    running_path.mkdir(parents=True, exist_ok=True)

logs_path = this_directory / f"./logs/{exp_id}"
logs_path.mkdir(parents=True, exist_ok=True)

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
writer = SummaryWriter(
    logs_path/'tensorboard',
)
#    filename_suffix='_' + exp_id)


save_head_to = lambda epoch: (
    running_path / f"{model_name}_{epoch}.pth"
).as_posix()  # head保存的位置


# %%
# 加载数据集
from torch.utils.data import DataLoader, ConcatDataset

model_require_input_image_size = [384, 512]     # Zoe
# model_require_input_image_size = [384, 384]   # Omni

dataset1 = CustomDataset(dataset_path_rgb1, dataset_path_depth1, image_size=model_require_input_image_size)
dataset2 = CustomDataset(dataset_path_rgb2, dataset_path_depth2, image_size=model_require_input_image_size)
# dataset = dataset2  # taskonomy
dataset = ConcatDataset([dataset1, dataset2])
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
import models

model = models.get_zoe_single_head_with_omni(pretrained_weights_path)
# model = models.ThreeDPT(pretrained_weights_path)
# model = models.OmniScale(pretrained_weights_path, head=models.MyNetwork_large())
# model = models.OmniScale(pretrained_weights_path, head=models.ResNet_v2())
model = model.to(device)
# model = nn.DataParallel(model)
# model.core.core = torch.compile(model.core.core)

# %%
# pretrained_head = system_data_path/'runs/最激进'/'ThreeDPT_20459.pth' # 最新一轮的保存好的结果
pretrained_head = system_data_path/'runs'/'最激进-Zoe/ZoeDepth_Omni_20560.pth' # 最新一轮的保存好的结果
checkpoint = torch.load(pretrained_head)
model.load_state_dict(checkpoint)
# %%
# from
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
from losses import ValidatedLoss, CompetitionLoss, REL

criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1, upper=20)
# criterion = ValidatedLoss(basic_loss=REL(), lower=0.1, upper=20)
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)
import sam.sam as sam
from sam.example.utility.bypass_bn import disable_running_stats, enable_running_stats

# optimizer = optim.AdamW(model.parameters(), lr=lr)
base_optimizer = torch.optim.AdamW
# base_optimizer = torch.optim.SGD
optimizer = sam.SAM(model.parameters(), base_optimizer, 
                    lr=lr, 
                    # momentum=0.9, 
                    weight_decay=0.01, 
                    # adaptive=False, rho=0.05)
                    adaptive=True, rho=2.0)
from torch.optim.lr_scheduler  import ExponentialLR, MultiStepLR, CosineAnnealingWarmRestarts
import torch.optim as optim
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
# scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# scheduler = CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=10, T_mult=2, eta_min=1e-5)

lrs = [l['lr'] for l in optimizer.base_optimizer.param_groups]
scheduler = optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 
                                          lrs, epochs=num_epochs, 
                                          steps_per_epoch=len(train_data_loader),
                                        cycle_momentum=True,
                                        base_momentum=0.85, max_momentum=0.95, 
                                        div_factor=1, 
                                        final_div_factor=10000, 
                                        pct_start=0.7, 
                                        three_phase=False)
# %%
# 训练网络
# def post_process(output):


import tqdm
bar = tqdm.tqdm(range(num_epochs), colour="green", leave=False, position=0)
for epoch in bar:
    inner_bar = tqdm.tqdm(train_data_loader, colour="yellow", leave=False, position=1)
    epoch_loss_sum = 0
    i_log = 0
    for images, depths_gt in inner_bar:
        # 似乎是to之后会导致device问题
        images = images.to(device)
        depths_gt = depths_gt.to(device)
        # debug
        # print(images.shape, depths_gt.shape)

        b, c, h, w = images.size()
        
        # SAM
        # first forward-backward pass
        enable_running_stats(model)
        output = model(images)
        pred_depths = output["metric_depth"].squeeze()
        loss = criterion(
            # depths_gt, pred_depths
            input=pred_depths, target=depths_gt # 必须pred在前，true灾后
        )  # use this loss for any training statistics
        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad() #zero_grad不能在backward后！
        optimizer.first_step(zero_grad=True)
        # grad_norm = np.array([p.norm().item()
        #     for p in head.parameters()
        # ]).mean()
        inner_bar.set_postfix(
            loss=loss.item(),
            #   grad_norm=grad_norm
        )
        epoch_loss_sum += loss.item()
        bar.set_postfix(Epoch=epoch, loss=epoch_loss_sum / (i_log + 1))
        writer.add_scalar(f"loss_{exp_id}", epoch_loss_sum / (i_log + 1), epoch * len(train_data_loader) + i_log)

        # second forward-backward pass
        disable_running_stats(model) 
        criterion(
            input=model(images)["metric_depth"].squeeze(), target=depths_gt
        ).backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)

        if i_log % save_steps == 0:
            torch.save(
                model.state_dict(), save_head_to(epoch * len(train_data_loader) + i_log)
            )
            
            retain_latest_models(running_path, model_name, num_to_retain=3)
            
        from PIL import Image
        import matplotlib.pyplot as plt

        if i_log % log_steps == 0:
            plt.imsave(
                (logs_path / "absolute_depth_map_test.png").as_posix(),
                pred_depths[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            plt.imsave(
                (logs_path / "ground_truth_test.png").as_posix(),
                depths_gt[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            from scipy import stats
            print(stats.describe(pred_depths[0].detach().cpu().squeeze().reshape(-1).numpy()))
            print(stats.describe(depths_gt[0].detach().cpu().squeeze().reshape(-1).numpy()))


        i_log += 1
        scheduler.step() # ZoeDepth也是写在batch里面的，跟随它。
        # writer.add_scalar(f"learning_rate_{exp_id}", epoch_loss_sum / (i_log + 1), epoch * len(train_data_loader) + i_log)

    # if epoch%save_steps == 0:
    #     torch.save(model.state_dict(), save_head_to(epoch))
    # scheduler1.step()
    # scheduler2.step()

# %%
writer.close()

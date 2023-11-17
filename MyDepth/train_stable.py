from boilerplate import *
# %%
# 定义数据路径
# exp_id = "复现实验"
# exp_id = "不使用PEFT-不使用lr_schedule-仅训练两轮"
exp_id = "复现初赛"
# model_name = "ZoeDepth_Omni"
# model_name = "ThreeDPT"
# model_name = "ThreeDPT"
model_name = "OmniScale"

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

# model_require_input_image_size = [384, 512]
model_require_input_image_size = [384, 384]

dataset1 = CustomDataset(dataset_path_rgb1, dataset_path_depth1, image_size=model_require_input_image_size)
dataset2 = CustomDataset(dataset_path_rgb2, dataset_path_depth2, image_size=model_require_input_image_size)
dataset = dataset2  # taskonomy
# dataset = ConcatDataset([dataset1, dataset2])
train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
import models

# model = models.get_zoe_single_head_with_omni(pretrained_weights_path)
# model = models.ThreeDPT(pretrained_weights_path)
model = models.OmniScale(pretrained_weights_path, head=models.MyNetwork_large())
model = model.to(device)
# model = nn.DataParallel(model)
# model.core.core = torch.compile(model.core.core)

# %%
# checkpoint = torch.load("/data/projects/depth/runs/复现实验" + "/ZoeDepth_Omni_660.pth")
# model.load_state_dict(checkpoint)
# model = nn.DataParallel(model)
# %%
# from
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
from losses import ValidatedLoss, CompetitionLoss, REL

# criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1, upper=20)
criterion = ValidatedLoss(basic_loss=REL(), lower=0.1, upper=20)
criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)
# import sam.sam as sam

optimizer = optim.AdamW(model.parameters(), lr=lr)
# base_optimizer = torch.optim.AdamW
# base_optimizer = torch.optim.SGD/
# optimizer = sam.SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
# optimizer = sam.SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)
# from torch.optim.lr_scheduler  import ExponentialLR, MultiStepLR, CosineAnnealingWarmRestarts
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
# scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
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
        output = model(images)

        pred_depths = output["metric_depth"]

        # SAM
        # first forward-backward pass
        loss = criterion(
            # depths_gt, pred_depths
            pred_depths, depths_gt # 必须pred在前，true灾后
        )  # use this loss for any training statistics
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() #zero_grad不能在backward后！
        # optimizer.first_step(zero_grad=True)
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
        # criterion(
        #     depths_gt, model(images)["metric_depth"]
        # ).backward()  # make sure to do a full forward pass
        # optimizer.second_step(zero_grad=True)

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
        # scheduler.step()
        # writer.add_scalar(f"learning_rate_{exp_id}", epoch_loss_sum / (i_log + 1), epoch * len(train_data_loader) + i_log)

    # if epoch%save_steps == 0:
    #     torch.save(model.state_dict(), save_head_to(epoch))
    # scheduler1.step()
    # scheduler2.step()

# %%
writer.close()

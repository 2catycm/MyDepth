#%%
from boilerplate import *
# batch_size*=24564/8478*24564/7338
# batch_size*=24564/15032
batch_size=int(batch_size)
batch_size = 1 # 为了看每一个图片的输出
print(f"batch_size={batch_size}")
# %%
# 定义数据路径
# exp_id = "复现实验"
# exp_id = "不使用PEFT-不使用lr_schedule-仅训练两轮"
# exp_id = "复现初赛"
# exp_id = "3DPT稳定版"
# exp_id = "3DPT稳定版+根据鱼眼做进一步微调"
# exp_id = "最激进-DPT-鱼眼优化"
exp_id = "最激进-Zoe"
# model_name = "ZoeDepth_Omni"
model_name = "ThreeDPT"
# model_name = "ThreeDPT"  
# model_name = "OmniScale"

logs_path = this_directory / f"./evals/{exp_id}"
logs_path.mkdir(parents=True, exist_ok=True)

# %%
# 加载数据集
from torch.utils.data import DataLoader, ConcatDataset

# ZoeDepth
# model_require_input_image_size = [384, 512]

# Omni
model_require_input_image_size = [384, 384]

# dataset1 = CustomDataset(dataset_path_rgb1, dataset_path_depth1, 
#                          image_size=model_require_input_image_size,
#                         #  do_fisheye_transform=True)
#                          do_fisheye_transform=False)
# dataset2 = CustomDataset(dataset_path_rgb2, dataset_path_depth2, 
#                          image_size=model_require_input_image_size,
#                         #  do_fisheye_transform=True)
#                          do_fisheye_transform=False)
dataset3 = CustomDataset('/data/projects/depth/5.跨场景单目深度估计/final_sample/test/',
                             '/data/projects/depth/5.跨场景单目深度估计/final_sample/ground_truth/',
                         image_size=model_require_input_image_size, 
                         replace_dict = {".jpg":".png"}, 
                         do_fisheye_transform=False)
# dataset = dataset2  # taskonomy
# dataset = ConcatDataset([dataset1, dataset2])
dataset = dataset3
val_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
import models

# model = models.get_zoe_single_head_with_omni(pretrained_weights_path)
model = models.ThreeDPT(pretrained_weights_path)
# model = models.OmniScale(pretrained_weights_path, head=models.MyNetwork_large())
model = model.to(device)
# model = nn.DataParallel(model)
# model.core.core = torch.compile(model.core.core)

# %%
# pretrained_head = system_data_path/'runs/最激进'/'ThreeDPT_20459.pth' # 最新一轮的保存好的结果（无鱼眼）
# pretrained_head = system_data_path/'runs/3DPT稳定版-根据鱼眼做进一步微调'/'ThreeDPT_17120.pth' # 最新一轮的保存好的结果
# pretrained_head = system_data_path/'runs/最激进-DPT-鱼眼优化'/'ThreeDPT_34080.pth' # 最新一轮的保存好的结果
# pretrained_head = system_data_path/'runs'/'valuable/最激进-DPT-鱼眼优化/ThreeDPT_37880.pth' # 最新一轮的保存好的结果
# pretrained_head = system_data_path/'runs'/'最激进-Zoe/ZoeDepth_Omni_20560.pth' # 最新一轮的Zoe
pretrained_head = system_data_path/'runs/最激进-DPT-鱼眼优化'/'ThreeDPT_44000.pth'
checkpoint = torch.load(pretrained_head)
model.load_state_dict(checkpoint)
# %%
# from
# criterion = nn.L1Loss()
# criterion = nn.MSELoss()
from losses import ValidatedLoss, CompetitionLoss, REL, SI_RMSE, compute

# criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1, upper=20)
# def new_compute(pred, true):
#     metrics = compute(true, pred).reshape(-1)
#     return 0.5*(metrics[0]+metrics[3])
# criterion = new_compute
criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.1*1000, upper=20*1000)
# criterion = ValidatedLoss(basic_loss=SI_RMSE(), lower=0.1, upper=20)
# criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0, upper=20)
# criterion = ValidatedLoss(basic_loss=REL(), lower=0, upper=20)
# criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.00001, upper=20)
# criterion = ValidatedLoss(basic_loss=CompetitionLoss(), lower=0.01, upper=20)
# criterion = ValidatedLoss(basic_loss=REL(), 
                        #   lower=0.1, upper=20)
# criterion = criterion.to(device)
# criterion = nn.DataParallel(criterion)
# import sam.sam as sam

model.eval() # 必须有
import tqdm

# inner_bar = enumerate(val_data_loader)
inner_bar = tqdm.tqdm(enumerate(val_data_loader), colour="yellow", leave=False, position=1)
epoch_loss_sum = 0
i_log = 0
log_steps = 1
with torch.no_grad():
    for i_log, (images, depths_gt) in inner_bar:
        # 似乎是to之后会导致device问题
        images = images.to(device)
        depths_gt = depths_gt.to(device)*512/1000
        # depths_gt = depths_gt.to(device)/8/4*10
        # depths_gt = depths_gt.to(device)/1000
        # debug
        # print(images.shape, depths_gt.shape)
        b, c, h, w = images.size()
        output = model(images)
        pred_depths = output["metric_depth"].squeeze(dim=1)

        # SAM
        # first forward-backward pass
        loss = criterion(
            # input=pred_depths, target=depths_gt # 必须pred在前，true灾后
            input=pred_depths*1000, target=depths_gt*1000 # 必须pred在前，true灾后
        )  # use this loss for any training statistics

        inner_bar.set_postfix(
            loss=loss.item(),
        )
        print(f"step_{i_log}_loss={loss.item()}") # 打印每一张图片的loss，方便debug
        epoch_loss_sum += loss.item()

        from PIL import Image
        import matplotlib.pyplot as plt
        if i_log % log_steps == 0:
            plt.imsave(
                (logs_path / f"y_pred{i_log}.png").as_posix(),
                pred_depths[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            plt.imsave(
                (logs_path / f"y_true{i_log}.png").as_posix(),
                depths_gt[0].detach().cpu().squeeze(),
                cmap="viridis",
            )
            # plt.imsave(
            #     (logs_path / f"x{i_log}.png").as_posix(),
            #     images[0].detach().cpu().squeeze(),
            #     cmap="viridis",
            # )
            from scipy import stats
            print("pred=", stats.describe(tensor_to_numpy(pred_depths[0]).reshape(-1)))
            print("true=", stats.describe(tensor_to_numpy(depths_gt[0]).reshape(-1)))
            # from scipy.special import rel_entr
            # print("KL Divergency=", sum(rel_entr(tensor_to_numpy(pred_depths[0]).reshape(-1),
            #       tensor_to_numpy(depths_gt[0]).reshape(-1))))

print("Overall Loss")
print(epoch_loss_sum//len(val_data_loader))



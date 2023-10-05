#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
import sys
sys.path.append((this_directory.parent/'ZoeDepth').as_posix())
sys.path.append((this_directory.parent).as_posix())
sys.path.append((this_directory.parent/"MyDepth").as_posix())
project_directory = this_directory.parent
#%%
from PIL import Image
import torchvision.transforms as transforms

test_img = Image.open("preliminary_a/img_100001.jpg")
transforms.ToTensor()(test_img).shape
# 720, 1280
#%%
from tqdm import tqdm
trans = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((720, 1280)),
    transforms.ToPILImage(),
])
image_paths = list(Path("result_1000").glob('*.png'))
bar = tqdm(image_paths)
for image_path in bar:
    image = Image.open(image_path)
    image = trans(image)
    image.save(image_path)
# %%
test_img= Image.open("result_1000/img_100001.png")
test_img= Image.open("submit_sample_a/img_100001.png")
transforms.ToTensor()(test_img).shape

# %%

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


posed_img_path = "./pose_set"
genuine_img_path = "./genuine_set"
lm_posed_path = "./lm_posed"
lm_genuine_path = "./lm_genuine"
lm_img_posed_path = "./lm_image_posed"
lm_img_genuine_path = "./lm_image_genuine"

label_map = {
    "GH": 1,               # Genuine happiness.
    "GS": 2,               # Genuine sad.
    "GD": 3,               # Genuine disgust.
    "GA": 4,               # Genuine anger.
    "GSur": 5,           # Genuine surprise.
    "PH": 6,             # Posed happiness.
    "PS": 7,             # Posed sadness.
    "PD": 8,             # Posed disgust.
    "PA": 9,             # Posed anger.
    "PSur": 10          # Posed surprise.
}


class CustomDataset(Dataset):
    def __init__(self, lm_img_path, target_img_path, target_lm_path,
                 target_img_transform=None, input_transform=None):
        self.input_lm_path = lm_img_path
        self.target_img_path = target_img_path
        self.target_lm_path = target_lm_path
        self.target_img_transform = target_img_transform
        self.input_transform = input_transform

    def __len__(self):
        return countFileNum(self.input_lm_path)

    def __getitem__(self, idx):
        input_file = os.listdir(self.input_lm_path)[idx]
        expression_type = removeNumDigit(input_file.split("_")[1].strip(".jpg"))
        # Get the image
        input_img = Image.open(os.path.join(self.input_lm_path, input_file))
        original_size = input_img.size[0]
        # Generate pose/genuine label (10 dimensional one-hot vector)
        PG_label = torch.zeros((10,))
        PG_label[label_map[expression_type] - 1] = 1

        # Generate target_img
        target_filename = input_file.replace("G", "P") if input_file.split("_")[1].startswith("G") else input_file.replace("P", "G")
        target_img = Image.open(os.path.join(self.target_img_path, target_filename))

        # Get target landmark coordinate
        df = pd.read_csv(os.path.join(self.target_lm_path, target_filename.strip(".jpg") + ".csv").replace("\\", "/"))
        lm_array = df.to_numpy()
        lm_array = torch.tensor(lm_array)

        if self.input_transform:
            input_img = self.input_transform(input_img)
        if self.target_img_transform:
            target_img = self.target_img_transform(target_img)

        return input_img, PG_label, target_img, lm_array, original_size


def countFileNum(path):

    counter = 0
    for file in os.listdir(path):
        counter += 1

    return counter


def removeNumDigit(str):
    return ''.join([i for i in str if not i.isdigit()])


# # Test dataset
# dataset = CustomDataset(lm_img_path=lm_img_posed_path,
#                         target_img_path=genuine_img_path,
#                         target_lm_path=lm_genuine_path)
#
#
#
# for i in range(len(dataset)):
#     input_img, PG_label, target_img, lm_array, original_size = dataset[i]
#
#     print(i, PG_label.size(), lm_array.size())
#
#     plt.subplots(1, 8)
#     plt.imshow(input_img)
#     plt.subplots(1, 8)
#     plt.imshow(target_img)
#
#     if i == 1:
#         plt.show()
#         break

# # test dataset.
# # Transformation on data.
# transform = transforms.Compose([
#     # resize image.
#     transforms.Resize([256, 256]),
#     # transform data into Tensor
#     transforms.ToTensor(),
#     # Normalize data into range(-1, 1)
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
# train_set = CustomDataset(lm_img_path=lm_img_posed_path, target_img_path=genuine_img_path,
#                                       target_lm_path=lm_genuine_path, input_transform=transform, target_img_transform=transform)
# trainloader = DataLoader(train_set, batch_size=64, shuffle=True)
# for batch_idx, (input_img, PG_label, target_img, lm_array, original_size) in enumerate(trainloader):
#     print(PG_label.size())
#     break
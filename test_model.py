import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from generator import Pix2PixGenerator
from dataset import CustomDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

posed_img_path = "./pose_set_test"
genuine_img_path = "./genuine_set"
lm_posed_path = "./lm_posed"
lm_genuine_path = "./lm_genuine"
lm_img_posed_path = "./lm_pose_img_test"
lm_img_genuine_path = "./lm_image_genuine"

checkpoint_path = "./20_checkpoint.pth"

transform = transforms.Compose([
        # resize image.
        transforms.Resize([256, 256]),
        # transform data into Tensor
        transforms.ToTensor()
    ])
transform_output = transforms.ToPILImage()

# define dataset and dataloader
train_set = CustomDataset(lm_img_path=lm_img_posed_path, target_img_path=genuine_img_path,
                                      target_lm_path=lm_genuine_path, input_transform=transform, target_img_transform=transform)
trainloader = DataLoader(train_set, batch_size=1, shuffle=False)

# load trained model
model = Pix2PixGenerator().to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['G_state_dict'])

# testing
for batch_idx, (input_img, PG_label, target_img, lm_array, original_size) in enumerate(trainloader):
    input_img, PG_label = input_img.to(device), PG_label.to(device)
    print(PG_label.shape)
    model.assign_label(PG_label)
    output = model(input_img)
    output = torch.squeeze(output, 0)
    output = transform_output(output)
    print(output.size)
    print(output)
    plt.imshow(output)
    plt.show()

import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import dataset
import generator
import discriminator
import loss

# sys.stdout = open("log_01.txt", "w")
posed_img_path = "./pose_set"
genuine_img_path = "./genuine_set"
lm_posed_path = "./lm_posed"
lm_genuine_path = "./lm_genuine"
lm_img_posed_path = "./lm_image_posed"
lm_img_genuine_path = "./lm_image_genuine"


def train(args, G, D, device, trainloader, optimizer_G, optimizer_D, epoch):
    print("********* Train Epoch " + str(epoch) + " start here *********")
    start = time.time()
    D.train()
    G.train()

    D_loss = 0.0
    G_loss = 0.0

    gan_loss = loss.GANLoss().to(device)
    per_loss = loss.VGGPerceptualLoss().to(device)
    id_loss = loss.IdentityLoss().to(device)
    re_loss = nn.L1Loss().to(device)

    for batch_idx, (input_img, PG_label, target_img, lm_array, original_size) in enumerate(trainloader):
        input_img, PG_label, target_img, lm_array = input_img.to(device), PG_label.to(device), target_img.to(device), lm_array.to(device)
        # pass in the label.
        G.assign_label(PG_label)
        # First train discriminator.
        target_img_fake = G(input_img)
        D_real = D(input_img, target_img)
        D_real_loss = gan_loss(D_real, target_is_real=True)
        D_fake = D(input_img, target_img_fake.detach())
        D_fake_loss = gan_loss(D_fake, target_is_real=False)
        D_train_loss = (D_real_loss + D_fake_loss) / 2

        optimizer_D.zero_grad()
        D_train_loss.backward()
        optimizer_D.step()

        # Then train generator.
        # perceptual loss
        G_per_loss = per_loss(target_img_fake, target_img)
        G_id_loss = id_loss(target_img_fake, target_img)
        G_re_loss = re_loss(target_img_fake, target_img)
        G_train_loss = args.lambda_per * G_per_loss + args.lambda_id * G_id_loss + args.lambda_re * G_re_loss

        optimizer_G.zero_grad()
        G_train_loss.backward()
        optimizer_G.step()

        D_loss += D_train_loss.item()
        G_loss += G_train_loss.item()

        # Print the log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD_loss: {:.6f}\tG_loss: {:.6f}'.format(
                epoch, batch_idx * len(input_img), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), D_train_loss.item(), G_train_loss.item()))

    D_loss /= len(trainloader.dataset)
    G_loss /= len(trainloader.dataset)

    end = time.time()
    print('\nTrain Epoch {} finished\nAverage Discriminator loss : {:.6f}, G loss: {:.6f}, time: {:.6f}'.format(
        epoch, D_loss, G_loss, end-start
    ))

    # save model
    checkpoint_path = './{}_checkpoint.pth'.format(epoch)
    torch.save({
        'G_state_dict': G.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)

    #sys.stdout.flush()


def main():
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description="Genuine/Posed facial expression conversion")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training. default=64')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing. default = 1000')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train. default = 14')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--betas', default=(0.9, 0.999),
                        help='betas in Adam optimizer.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--lambda_per', type=int, default=10, metavar='M',
                        help='weight parameter for perceptual loss in generator')
    parser.add_argument('--lambda_id', type=int, default=1, metavar='M',
                        help='weight parameter for identity loss in generator')
    parser.add_argument('--lambda_re', type=int, default=100, metavar='M',
                        help='weight parameter for reconstruction loss in generator')
    parser.add_argument('--lambda_gan', type=int, default=10, metavar='M',
                        help='weight parameter for gan loss in generator')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    print("cuda is available on this machine :" + str(use_cuda))

    device = torch.device("cuda" if use_cuda else "cpu")

    #### First train from posed image to genuine image.

    # For both target and input image, only normalize them into [-1, 1]
    # Transformation on data.
    transform = transforms.Compose([
        # resize image.
        transforms.Resize([256, 256]),
        # transform data into Tensor
        transforms.ToTensor(),
        # Normalize data into range(-1, 1)
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Define dataset.
    train_set = dataset.CustomDataset(lm_img_path=lm_img_posed_path, target_img_path=genuine_img_path,
                                      target_lm_path=lm_genuine_path, input_transform=transform, target_img_transform=transform)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # testloader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)
    # valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # define model (generator and discriminator)
    G = generator.Pix2PixGenerator(label=torch.zeros((args.batch_size, 10))).to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    D = discriminator.Discriminator().to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    # Learning rate decay.
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=args.gamma)
    scheduler_D = StepLR(optimizer_G, step_size=50, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, G, D, device, trainloader, optimizer_G, optimizer_D, epoch)
        scheduler_G.step()
        scheduler_D.step()


    #### Then train from genuine image to posed image.
    train_set = dataset.CustomDataset(lm_img_path=lm_img_genuine_path, target_img_path=posed_img_path,
                                      target_lm_path=lm_posed_path, input_transform=transform,
                                      target_img_transform=transform)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(50, args.epochs + 50):
        train(args, G, D, device, trainloader, optimizer_G, optimizer_D, epoch)
        scheduler_G.step()
        scheduler_D.step()


    end = time.time()
    print("Total training time is: "+str(end-start))
    # sys.stdout.close()

# Training
main()


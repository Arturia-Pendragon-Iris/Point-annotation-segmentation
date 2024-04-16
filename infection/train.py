import argparse, os
import torch
import math, random
from torchvision.ops import sigmoid_focal_loss
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from infection.utils import TrainSetLoader_2D, TrainSetLoader_3D
from infection.losses import dice_loss, ce_loss, global_ce, rsc_loss
from infection.model import SegNet_2D, SegNet_3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=3, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.6
                    , help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    cuda = opt.cuda
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = SegNet_2D(in_ch=1, final_ch=4)
    # model.load_state_dict(torch.load("/data/Train_and_Test/infection_new.pth"))
    model = model.to('cuda')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)
    # exit()
    print("===> Setting Optimizer")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        data_set = TrainSetLoader_2D(device)
        data_loader = DataLoader(dataset=data_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
        trainor(data_loader, optimizer, model, epoch)
        scheduler.step()
        # seg_scheduler.step()


def trainor(data_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    for iteration, (raw, gt, in_blank, out_blank, hessian) in enumerate(data_loader):
        # print(raw.shape, mask.shape)
        pre, F = model(raw)

        loss_ce = global_ce(pre, raw, in_blank, out_blank)
        loss_rcs = rsc_loss(F, torch.concatenate((pre, hessian), dim=1), in_blank, out_blank)
        print(loss_ce, loss_rcs)
        loss = loss_ce + loss_rcs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = dice_loss(pre, gt)
        loss_epoch += dice

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, dice, loss_epoch / (iteration % 100 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/data/Train_and_Test/infection_new/PCE")
            # save_checkpoint(seg_model, epoch, "/home/chuy/Artery_Vein_Upsampling/checkpoint/whole/segment/")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "test_1.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


train()



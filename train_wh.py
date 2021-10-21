import torch
from data_loader import MVTecDRAEMTrainDataset
from dataloader_wh import Mydatatest, Mydatatrain
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    save_checkpoint = os.path.join(args.checkpoint_path, args.train_name)
    os.makedirs(save_checkpoint, exist_ok=True)

    run_name = args.train_name + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs)
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": args.lr},
                                  {"params": model_seg.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs*0.8, args.epochs*0.9], gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    #dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
    dataset = Mydatatrain(args.train_list, args.anomaly_list, resize_shape=[256, 256])



    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=True, num_workers=16)

    n_iter = 0
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()
            #import pdb;pdb.set_trace()
            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec, gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()



            if args.visualize and n_iter % 200 == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
            if args.visualize and n_iter % 400 == 0:
                t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')

            n_iter +=1
        scheduler.step()
        if epoch % 20 == 1:

            print(f'l2loss{l2_loss}   ssim_loss{ssim_loss}   segment_loss{segment_loss}')



            torch.save(model.state_dict(), os.path.join(save_checkpoint, run_name + str(epoch)+'.pth'))
            torch.save(model_seg.state_dict(), os.path.join(save_checkpoint, run_name + str(epoch)+'_seg.pth'))

        #torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
        #torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_name', action='store', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--anomaly_list', type=str, required=True)
    parser.add_argument('--test_list', type=str, required=True)

    args = parser.parse_args()
    #
    # obj_batch = [['capsule'],
    #              ['bottle'],
    #              ['carpet'],
    #              ['leather'],
    #              ['pill'],
    #              ['transistor'],
    #              ['tile'],
    #              ['cable'],
    #              ['zipper'],
    #              ['toothbrush'],
    #              ['metal_nut'],
    #              ['hazelnut'],
    #              ['screw'],
    #              ['grid'],
    #              ['wood']
    #              ]
    #
    # if int(args.obj_id) == -1:
    #     obj_list = ['capsule',
    #                  'bottle',
    #                  'carpet',
    #                  'leather',
    #                  'pill',
    #                  'transistor',
    #                  'tile',
    #                  'cable',
    #                  'zipper',
    #                  'toothbrush',
    #                  'metal_nut',
    #                  'hazelnut',
    #                  'screw',
    #                  'grid',
    #                  'wood'
    #                  ]
    #     picked_classes = obj_list
    # else:
    #     picked_classes = obj_batch[int(args.obj_id)]
    #
    # with torch.cuda.device(args.gpu_id):
    train_on_device(args)


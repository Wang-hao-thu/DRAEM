import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os

from dataloader_wh import Mydatatest, Mydatatrain

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def write_results_to_file_wh(args, anomaly_score_gt, anomaly_score_prediction, img_pathes):
    f1 = open(args.save_file, 'w')
    for i in range(len(anomaly_score_prediction)):
        img_path = img_pathes[i][0]
        label = int(anomaly_score_gt[i])
        score = float(anomaly_score_prediction[i])
        f1.write(img_path + ' ' + str(label) +' ' + str(score) + '\n')




def test(args):
    img_dim = 256
    run_name = args.test_name

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(args.rec_checkpoint))#, map_location='cuda:0')
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(args.seg_checkpoint))#, map_location='cuda:0')
    model_seg.cuda()
    model_seg.eval()

    dataset = Mydatatest(args.test_list, resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))

    anomaly_score_gt = []
    anomaly_score_prediction = []
    img_pathes = []

    # display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    # display_gt_images = torch.zeros((16,3 ,256 ,256)).cuda()
    # display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # cnt_display = 0
    display_indices = np.random.randint(len(dataloader), size=(16,))

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()
        img_path = sample_batched['image_path']

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        #true_mask = sample_batched["mask"]
        #true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        #out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)
        img_pathes.append(img_path)


    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)

    write_results_to_file_wh(args, anomaly_score_gt, anomaly_score_prediction, img_pathes)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--rec_checkpoint', type=str, required=True)
    parser.add_argument('--seg_checkpoint', type=str, required=True)
    parser.add_argument('--test_name', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)


    args = parser.parse_args()


    with torch.cuda.device(args.gpu_id):
        test(args)

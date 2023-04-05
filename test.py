import torch
import numpy as np
from PIL import Image
import os

from models import OFSRCNN, FSRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, ssim

def test(scale, model_name, dataset_name):

    weight_dir = os.path.join('./outputs/', '{}_X{}_best.pth'.format(model_name, scale))
    hr_dir = './Datasets/' + dataset_name + '/HR'
    out_dir = './Datasets/' + dataset_name + '/' + model_name

    device = torch.device('cpu')
    model = FSRCNN(scale=scale).to(device) if model_name == 'FSRCNN' else OFSRCNN(scale=scale).to(device)
    model.load_state_dict(torch.load(weight_dir))

    model.eval()

    if not os.path.exists(os.path.join(out_dir, 'X{}'.format(scale))):
        os.makedirs(os.path.join(out_dir, 'X{}'.format(scale)))

    for img in sorted(os.listdir(hr_dir)):
        image = Image.open(os.path.join(hr_dir, img)).convert('RGB')
        image_lr = image.resize((image.width // scale, image.height // scale), resample=Image.BICUBIC)

        # image to ycbcr arr
        image_arr = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image_arr)
        # y of ycbcr
        y = ycbcr[..., 0]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0) # dim expand

        # lr image to ycbcr arr
        image_lr = np.array(image_lr).astype(np.float32)
        ycbcr_lr = convert_rgb_to_ycbcr(image_lr)
        # y of lr ycbcr
        y_lr = ycbcr_lr[..., 0]
        y_lr /= 255.
        y_lr = torch.from_numpy(y_lr).to(device)
        y_lr = y_lr.unsqueeze(0).unsqueeze(0) # dim expand

        with torch.no_grad():
            preds = model(y_lr).clamp(0.0, 1.0)

        # resize preds if necessary
        if preds.size()[2] != image.height or preds.size()[3] != image.width:
            temp_image = Image.fromarray(preds.numpy().squeeze(0).squeeze(0))
            temp_image = temp_image.resize((image.width, image.height), resample=Image.BICUBIC)
            temp_image_arr = np.array(temp_image).astype(np.float32)
            preds = torch.from_numpy(temp_image_arr).to(device).unsqueeze(0).unsqueeze(0)
        print(img)
        print('PSNR on y: {:.4f}'.format(calc_psnr(y, preds)))
        print('SSIM on y: {:.4f}'.format(ssim(y, preds)))

        preds = preds.mul(255.0).numpy().squeeze(0).squeeze(0)

        # (channels,imagesize,imagesize) to (imagesize,imagesize,channels)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]], dtype=object).transpose([1, 2, 0])

        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output)

        output.save(os.path.join(out_dir, 'X{}'.format(scale), img))
    print('test done')
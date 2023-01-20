import argparse
import os

from colorizers import *
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, help='Path to the pre-trained model (if any)', default=None)
parser.add_argument('--input-path', type=str, help='Path to folder of input images', default='imgs/grey/')
parser.add_argument('--output-path', type=str, help='Path to store the predicted images', default='imgs/output/')
parser.add_argument('--use-gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

# load siggraph colorizer
colorizer_siggraph17 = siggraph17(pretrained=True, local_pth=opt.model_path).eval()

if (opt.use_gpu):
    colorizer_siggraph17.cuda()

files = os.listdir(opt.input_path)
files = [f for f in files if f.endswith(".jpg")]

# get the prediction per image
for file in files:
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(os.path.join(opt.input_path, file))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # save the image
    os.makedirs(opt.output_path, exist_ok=True)

    # https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type
    image = Image.fromarray((out_img_siggraph17 * 255).astype(np.uint8))

    # remove the alpha channel (otherwise we can't save it as jpg)
    image = image.convert('RGB')
    image.save(os.path.join(opt.output_path, file))

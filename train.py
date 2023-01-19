import argparse
import os

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True)
colorizer_siggraph17 = siggraph17(pretrained=True)

if (opt.use_gpu):
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()


# load training data from folder
train_folder = './input/218/train/100/grey/'

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
train_imgs = []
for img in os.listdir(train_folder):
    img = Image.open(train_folder + img)
    img = np.array(img)

    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    train_imgs.append(tens_l_rs)

# Softmax loss
loss_fn_eccv16 = torch.nn.MSELoss()
loss_fn_siggraph17 = torch.nn.MSELoss()

# Optimizer
opt_eccv16 = torch.optim.Adam(colorizer_eccv16.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)
opt_siggraph17 = torch.optim.Adam(colorizer_siggraph17.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)

train_imgs = train_imgs[:3]

# Train loop
for epoch in range(1):
    for i, img in enumerate(train_imgs):
        # Forward pass
        out_eccv16 = colorizer_eccv16(img)
        out_siggraph17 = colorizer_siggraph17(img)

        # Loss
        loss_eccv16 = loss_fn_eccv16(out_eccv16, img)
        loss_siggraph17 = loss_fn_siggraph17(out_siggraph17, img)

        # Backward pass
        opt_eccv16.zero_grad()
        loss_eccv16.backward()
        opt_eccv16.step()

        opt_siggraph17.zero_grad()
        loss_siggraph17.backward()
        opt_siggraph17.step()

        # Rounded to 2 decimals
        print(f"Epoch: {epoch}, Image: {i}, Loss ECCV16: {round(loss_eccv16.item(), 2)}, Loss SIGGRAPH17: {round(loss_siggraph17.item(), 2)}")

# Save models to file
torch.save(colorizer_eccv16.state_dict(), 'colorizer_eccv16.pth')
torch.save(colorizer_siggraph17.state_dict(), 'colorizer_siggraph17.pth')

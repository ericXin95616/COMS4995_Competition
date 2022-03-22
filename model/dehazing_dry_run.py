import glob
import os
import cv2
from dehazing_BPPNet import Dataset, DataLoader, DU_Net, train
import PIL
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as F6


BPP_root = '../BPP_train'
path_of_train_hazy_images = os.path.join(BPP_root, 'haze_train')
path_of_train_clean_images = os.path.join(BPP_root, 'dehaze_train')

path_of_test_hazy_images = os.path.join(BPP_root, 'haze_test')
path_of_test_clean_images = os.path.join(BPP_root, 'dehaze_test')

path_of_result_images = os.path.join(BPP_root, 'result_dehaze')

images_paths_haze_train = glob.glob(path_of_train_hazy_images + '/*.jpg')
images_paths_clean_train = glob.glob(path_of_train_clean_images + '/*.jpg')
images_paths_haze_test = glob.glob(path_of_test_hazy_images + '/*.jpg')

train_dataset = Dataset(images_paths_haze_train, images_paths_clean_train, augment=False)
train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            shuffle=False
        )

input_unet_channel = 3
output_unet_channel = 3
input_dis_channel = 3
epochs = 150
DUNet = DU_Net(input_unet_channel, output_unet_channel, input_dis_channel).cuda()
# train(DUNet, train_loader, max_epochs=epochs)

# load the weights
path_of_generator_weight = './weight/generator1.pth'  # path where the weights of genertaor are stored
path_of_discriminator_weight = './weight/discriminator1.pth'  # path where the weights of discriminator are stored
DUNet.load(path_of_generator_weight, path_of_discriminator_weight)
# train(DUNet, train_loader, max_epochs=epochs)


def to_tensor(img):
    img_t = F6.to_tensor(img).float()
    return img_t


def postprocess(img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


def generate_dehaze_images(DUNet, test_haze_images_list):
    for i in range(len(test_haze_images_list)):
        haze_image = cv2.imread(test_haze_images_list[i])
        haze_image = PIL.Image.fromarray(haze_image)
        # haze_crop = haze_image.crop((10, 10, 650, 650))
        haze_image = haze_image.resize((256, 256), resample=PIL.Image.BICUBIC)
        haze_image = np.array(haze_image)
        haze_image = cv2.cvtColor(haze_image, cv2.COLOR_BGR2YCrCb)
        haze_image = to_tensor(haze_image).cuda()
        haze_image = haze_image.reshape(1, 3, 256, 256)

        dehaze_image = DUNet.predict(haze_image)

        dehaze_image = postprocess(dehaze_image)[0]
        dehaze_image = dehaze_image.cpu().detach().numpy()
        dehaze_image = dehaze_image.astype('uint8')
        dehaze_image = dehaze_image.reshape(256, 256, 3)
        dehaze_image = cv2.cvtColor(dehaze_image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(path_of_result_images + '/' + str(int(Path(test_haze_images_list[i]).stem)) + '.jpg', dehaze_image)


print(images_paths_haze_test)
generate_dehaze_images(DUNet, images_paths_haze_test)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
import PIL.Image as pil_image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class Dataset(object):
    def __init__(self, images_dir, patch_size, jpeg_quality, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            label = tf.read_file(self.image_files[idx])
            label = tf.image.decode_jpeg(label, channels=3)
            label = pil_image.fromarray(label.numpy())
        else:
            label = pil_image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, label.width - self.patch_size)
        crop_y = random.randint(0, label.height - self.patch_size)
        label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        # additive jpeg noise
        buffer = io.BytesIO()
        label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        input = pil_image.open(buffer)

        input = np.array(input).astype(np.float32)
        label = np.array(label).astype(np.float32)
        input = np.transpose(input, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)

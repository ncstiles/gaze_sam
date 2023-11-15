#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import os
import sys
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from trt_sam import SamPad, SamResize

# try:
#     from detectron2.config import get_cfg
# except ImportError:
#     print("Could not import Detectron 2 modules. Maybe you did not install Detectron 2")
#     print("Please install Detectron 2, check https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md")
#     sys.exit(1)

class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, encoder, max_num_images=None, exact_batches=False, config_file=None):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param config_file: The path pointing to the Detectron 2 yaml file which describes the model.
        """

        # Find images in the given input path.
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        print("num images:", self.num_images)

        # Handle Tensor Shape.
        self.shapes = {
            "image_embeddings": (1, 256, 64, 64), 
            "point_coords": (32, 1, 2), 
            "point_labels": (32, 1),
            "mask_input": (1, 1, 256, 256),
            "has_mask_input": (1,)
        }
        self.dtype = dtype
        self.batch_size = 1 # TODO: remove later
        # self.shape = shape
        # assert len(self.shape) == 4
        # self.batch_size = shape[0]
        # assert self.batch_size > 0
        # self.format = None
        # self.width = -1
        # self.height = -1
        # if self.shape[1] == 3:
        #     self.format = "NCHW"
        #     self.height = self.shape[2]
        #     self.width = self.shape[3]
        # elif self.shape[3] == 3:
        #     self.format = "NHWC"
        #     self.height = self.shape[1]
        #     self.width = self.shape[2]
        # assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed.
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0:self.num_images]

        # Subdivide the list of images into batches.
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices.
        self.image_index = 0
        self.batch_index = 0

        # Load encoder.
        self.encoder = encoder


    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """
        def preprocess(image):
            transform = transforms.Compose(
                [
                    SamResize(512),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                        std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                    ),
                    SamPad(512),
                ]
            )
            return transform(image).unsqueeze(dim=0).cuda()

        raw_image = np.asarray(Image.open(image_path).convert("RGB"))
        if raw_image.shape[0] * raw_image.shape[1] > 1280 * 720:
            raw_image = cv2.resize(raw_image, 1280, 720)
        image = preprocess(raw_image)

        return image
    
    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        print("dtype:", self.dtype)
        point_coords = np.array([[717, 254],[695, 256], [673, 259], [651, 261], [629, 264], [607, 266], [585, 268], [563, 271],
                                [541, 273], [519, 276], [497, 278], [475, 280], [453, 283], [431, 285], [409, 288], [387, 290],
                                [365, 292], [343, 295], [321, 297], [299, 299], [277, 302], [255, 304], [233, 307], [211, 309],
                                [189, 311], [167, 314], [145, 316], [123, 319], [101, 321], [ 79, 323], [ 57, 326], [ 35, 328]],
                                dtype=self.dtype)
        point_labels = np.ones((32,1), dtype=self.dtype)
        mask_input = np.zeros((1, 1, 256, 256), dtype=self.dtype)
        has_mask_input = np.zeros((1), dtype=self.dtype)

        print('point coords shape:', point_coords.shape)
        print('point label shape:', point_labels.shape)
        print('mask input shape:', mask_input.shape)
        print('has mask input:', has_mask_input.shape)


        for batch_images in self.batches:
            image_embeddings =  np.zeros((1, 254, 64, 64), dtype=self.dtype)
            print("NUM BATCH IMGS:", len(batch_images))

            for image in batch_images:
                self.image_index += 1
                print("about to preprocess img")
                preprocessed_image = self.preprocess_image(image)
                print("done preprocessing img, feeding into encoder")
                image_embeddings = self.encoder(preprocessed_image)
                print("done ran the encoder")
                print("img_embedding shape:", image_embeddings.shape)
                print(f'\tmin: {min(image_embeddings)} max: {max(image_embeddings)}')
            self.batch_index += 1
            yield {
                "image_embeddings": image_embeddings.detach().cpu().numpy(),
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input
            }

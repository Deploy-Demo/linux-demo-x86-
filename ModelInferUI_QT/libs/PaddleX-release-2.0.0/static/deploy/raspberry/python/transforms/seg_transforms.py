# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .ops import *
import random
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from collections import OrderedDict


class SegTransform:
    """ 分割transform基类
    """

    def __init__(self):
        pass


class Compose(SegTransform):
    """根据数据预处理/增强算子对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。

    Args:
        transforms (list): 数据预处理/增强算子。

    Raises:
        TypeError: transforms不是list对象
        ValueError: transforms元素个数小于1。

    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                            'must be equal or larger than 1!')
        self.transforms = transforms
        self.to_rgb = False

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (str/np.ndarray): 标注图像路径/标注图像np.ndarray数据。

        Returns:
            tuple: 根据网络所需字段所组成的tuple；字段由transforms中的最后一个数据预处理操作决定。
        """

        if im_info is None:
            im_info = list()
        if isinstance(im, np.ndarray):
            if len(im.shape) != 3:
                raise Exception(
                    "im should be 3-dimensions, but now is {}-dimensions".
                    format(len(im.shape)))
        else:
            try:
                im = cv2.imread(im).astype('float32')
            except:
                raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if label is not None:
            if not isinstance(label, np.ndarray):
                label = np.asarray(Image.open(label))
        for op in self.transforms:
            if isinstance(op, SegTransform):
                outputs = op(im, im_info, label)
                im = outputs[0]
                if len(outputs) >= 2:
                    im_info = outputs[1]
                if len(outputs) == 3:
                    label = outputs[2]
            else:
                im = execute_imgaug(op, im)
                if label is not None:
                    outputs = (im, im_info, label)
                else:
                    outputs = (im, im_info)
        return outputs

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        transform_names = [type(x).__name__ for x in self.transforms]
        for aug in augmenters:
            if type(aug).__name__ in transform_names:
                print(
                    "{} is already in ComposedTransforms, need to remove it from add_augmenters().".
                    format(type(aug).__name__))
        self.transforms = augmenters + self.transforms


class RandomHorizontalFlip(SegTransform):
    """以一定的概率对图像进行水平翻转。当存在标注图像时，则同步进行翻转。

    Args:
        prob (float): 随机水平翻转的概率。默认值为0.5。

    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if random.random() < self.prob:
            im = horizontal_flip(im)
            if label is not None:
                label = horizontal_flip(label)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomVerticalFlip(SegTransform):
    """以一定的概率对图像进行垂直翻转。当存在标注图像时，则同步进行翻转。

    Args:
        prob (float): 随机垂直翻转的概率。默认值为0.1。
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if random.random() < self.prob:
            im = vertical_flip(im)
            if label is not None:
                label = vertical_flip(label)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Resize(SegTransform):
    """调整图像大小（resize），当存在标注图像时，则同步进行处理。

    - 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
    - 当目标大小（target_size）类型为list或tuple时，根据插值方式，
      将图像resize为target_size, target_size的输入应为[w, h]或（w, h）。

    Args:
        target_size (int|list|tuple): 目标大小。
        interp (str): resize的插值方式，与opencv的插值方式对应，
            可选的值为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为"LINEAR"。

    Raises:
        TypeError: target_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
        AssertionError: interp的取值不在['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']之内。
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size, interp='LINEAR'):
        self.interp = interp
        assert interp in self.interp_dict, "interp should be one of {}".format(
            interp_dict.keys())
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info跟新字段为：
                    -shape_before_resize (tuple): 保存resize之前图像的形状(h, w）。

        Raises:
            ZeroDivisionError: im的短边为0。
            TypeError: im不是np.ndarray数据。
            ValueError: im不是3维nd.ndarray。
        """
        if im_info is None:
            im_info = OrderedDict()
        im_info.append(('resize', im.shape[:2]))

        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeImage: image type is not np.ndarray.")
        if len(im.shape) != 3:
            raise ValueError('ResizeImage: image is not 3-dimensional.')
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('ResizeImage: min size of image is 0')

        if isinstance(self.target_size, int):
            resize_w = self.target_size
            resize_h = self.target_size
        else:
            resize_w = self.target_size[0]
            resize_h = self.target_size[1]
        im_scale_x = float(resize_w) / float(im_shape[1])
        im_scale_y = float(resize_h) / float(im_shape[0])

        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp_dict[self.interp])
        if label is not None:
            label = cv2.resize(
                label,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp_dict['NEAREST'])
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeByLong(SegTransform):
    """对图像长边resize到固定值，短边按比例进行缩放。当存在标注图像时，则同步进行处理。

    Args:
        long_size (int): resize后图像的长边大小。
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info新增字段为：
                    -shape_before_resize (tuple): 保存resize之前图像的形状(h, w)。
        """
        if im_info is None:
            im_info = OrderedDict()

        im_info.append(('resize', im.shape[:2]))
        im = resize_long(im, self.long_size)
        if label is not None:
            label = resize_long(label, self.long_size, cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeByShort(SegTransform):
    """根据图像的短边调整图像大小（resize）。

    1. 获取图像的长边和短边长度。
    2. 根据短边与short_size的比例，计算长边的目标长度，
       此时高、宽的resize比例为short_size/原图短边长度。
    3. 如果max_size>0，调整resize比例：
       如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
    4. 根据调整大小的比例对图像进行resize。

    Args:
        target_size (int): 短边目标长度。默认为800。
        max_size (int): 长边目标长度的最大限制。默认为1333。

     Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, short_size=800, max_size=1333):
        self.max_size = int(max_size)
        if not isinstance(short_size, int):
            raise TypeError(
                "Type of short_size is invalid. Must be Integer, now is {}".
                format(type(short_size)))
        self.short_size = short_size
        if not (isinstance(self.max_size, int)):
            raise TypeError("max_size: input type is invalid.")

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                   存储与图像相关信息的字典和标注图像np.ndarray数据。
                   其中，im_info更新字段为：
                       -shape_before_resize (tuple): 保存resize之前图像的形状(h, w)。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = OrderedDict()
        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeByShort: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('ResizeByShort: image is not 3-dimensional.')
        im_info.append(('resize', im.shape[:2]))
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        scale = float(self.short_size) / im_short_size
        if self.max_size > 0 and np.round(scale *
                                          im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im = cv2.resize(
            im, (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST)
        if label is not None:
            im = cv2.resize(
                label, (resized_width, resized_height),
                interpolation=cv2.INTER_NEAREST)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeRangeScaling(SegTransform):
    """对图像长边随机resize到指定范围内，短边按比例进行缩放。当存在标注图像时，则同步进行处理。

    Args:
        min_value (int): 图像长边resize后的最小值。默认值400。
        max_value (int): 图像长边resize后的最大值。默认值600。

    Raises:
        ValueError: min_value大于max_value
    """

    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(min_value,
                                                              max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        im = resize_long(im, random_size, cv2.INTER_LINEAR)
        if label is not None:
            label = resize_long(label, random_size, cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeStepScaling(SegTransform):
    """对图像按照某一个比例resize，这个比例以scale_step_size为步长
    在[min_scale_factor, max_scale_factor]随机变动。当存在标注图像时，则同步进行处理。

    Args:
        min_scale_factor（float), resize最小尺度。默认值0.75。
        max_scale_factor (float), resize最大尺度。默认值1.25。
        scale_step_size (float), resize尺度范围间隔。默认值0.25。

    Raises:
        ValueError: min_scale_factor大于max_scale_factor
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]

        im = cv2.resize(
            im, (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(
                label, (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Normalize(SegTransform):
    """对图像进行标准化。
    1.尺度缩放到 [0,1]。
    2.对图像进行减均值除以标准差操作。

    Args:
        mean (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5]。
        std (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5]。

    Raises:
        ValueError: mean或std不是list对象。std包含0。
    """

    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Padding(SegTransform):
    """对图像或标注图像进行padding，padding方向为右和下。
    根据提供的值对图像或标注图像进行padding操作。

    Args:
        target_size (int|list|tuple): padding后图像的大小。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认值为255。

    Raises:
        TypeError: target_size不是int|list|tuple。
        ValueError:  target_size为list|tuple时元素个数不等于2。
    """

    def __init__(self,
                 target_size,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info新增字段为：
                    -shape_before_padding (tuple): 保存padding之前图像的形状(h, w）。

        Raises:
            ValueError: 输入图像im或label的形状大于目标值
        """
        if im_info is None:
            im_info = OrderedDict()
        im_info.append(('padding', im.shape[:2]))

        im_height, im_width = im.shape[0], im.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im = cv2.copyMakeBorder(
                im,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.label_padding_value)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomPaddingCrop(SegTransform):
    """对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作。

    Args:
        crop_size (int|list|tuple): 裁剪图像大小。默认为512。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认值为255。

    Raises:
        TypeError: crop_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
    """

    def __init__(self,
                 crop_size=512,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'when crop_size is list or tuple, it should include 2 elements, but it is {}'
                    .format(crop_size))
        elif not isinstance(crop_size, int):
            raise TypeError(
                "Type of crop_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return (im, im_info)
            else:
                return (im, im_info, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im = cv2.copyMakeBorder(
                    im,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                if label is not None:
                    label = cv2.copyMakeBorder(
                        label,
                        0,
                        pad_height,
                        0,
                        pad_width,
                        cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = im.shape[0]
                img_width = im.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(w_off + crop_width
                                                            ), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomBlur(SegTransform):
    """以一定的概率对图像进行高斯模糊。

    Args：
        prob (float): 图像模糊概率。默认为0.1。
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomScaleAspect(SegTransform):
    """裁剪并resize回原始尺寸的图像和标注图像。
    按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。

    Args：
        min_scale (float)：裁取图像占原始图像的面积比，取值[0，1]，为0时则返回原图。默认为0.5。
        aspect_ratio (float): 裁取图像的宽高比范围，非负值，为0时返回原图。默认为0.33。
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    label = label[h1:(h1 + dh), w1:(w1 + dw)]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    label = cv2.resize(
                        label, (img_width, img_height),
                        interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomDistort(SegTransform):
    """对图像进行随机失真。

    1. 对变换的操作顺序进行随机化操作。
    2. 按照1中的顺序以一定的概率对图像进行随机像素内容变换。

    Args:
        brightness_range (float): 明亮度因子的范围。默认为0.5。
        brightness_prob (float): 随机调整明亮度的概率。默认为0.5。
        contrast_range (float): 对比度因子的范围。默认为0.5。
        contrast_prob (float): 随机调整对比度的概率。默认为0.5。
        saturation_range (float): 饱和度因子的范围。默认为0.5。
        saturation_prob (float): 随机调整饱和度的概率。默认为0.5。
        hue_range (int): 色调因子的范围。默认为18。
        hue_prob (float): 随机调整色调的概率。默认为0.5。
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [brightness, contrast, saturation, hue]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob
        }
        for id in range(4):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ArrangeSegmenter(SegTransform):
    """获取训练/验证/预测所需的信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内
    """

    def __init__(self, mode):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, im, im_info, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当mode为'train'或'eval'时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当mode为'test'时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；当mode为
                'quant'时，返回的tuple为(im,)，为图像np.ndarray数据。
        """
        im = permute(im, False)
        if self.mode == 'train' or self.mode == 'eval':
            label = label[np.newaxis, :, :]
            return (im, label)
        elif self.mode == 'test':
            return (im, im_info)
        else:
            return (im, )


class ComposedSegTransforms(Compose):
    """ 语义分割模型(UNet/DeepLabv3p)的图像处理流程，具体如下
        训练阶段：
        1. 随机对图像以0.5的概率水平翻转
        2. 按不同的比例随机Resize原图
        3. 从原图中随机crop出大小为train_crop_size大小的子图，如若crop出来的图小于train_crop_size，则会将图padding到对应大小
        4. 图像归一化
        预测阶段：
        1. 图像归一化

        Args:
            mode(str): 图像处理所处阶段，训练/验证/预测，分别对应'train', 'eval', 'test'
            train_crop_size(list): 模型训练阶段，随机从原图crop的大小
            mean(list): 图像均值
            std(list): 图像方差
    """

    def __init__(self,
                 mode,
                 train_crop_size=[769, 769],
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5]):
        if mode == 'train':
            # 训练时的transforms，包含数据增强
            pass
        else:
            # 验证/预测时的transforms
            transforms = [Normalize(mean=mean, std=std)]

        super(ComposedSegTransforms, self).__init__(transforms)

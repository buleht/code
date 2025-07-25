import numpy as np
import struct
import os
import scipy.io
import time

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    def forward(self, input_layer, content_layer):
         # TODO： 计算风格迁移图像和目标内容图像的内容损失
         #确保风格迁移图像和目标内容图像的大小尺寸相同
         #reshape是自动重塑数组尺寸的函数，-1相当于让numpy自动去计算，在这里是让numpy将输入数组转换成一维数组
        assert input_layer.shape == content_layer.shape
        loss = 0.5 * np.sum((input_layer.reshape(-1) - content_layer.reshape(-1))**2) / input_layer.size
        return loss
    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        bottom_diff = (input_layer-content_layer)/input_layer.size
        return bottom_diff

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的Gram 矩阵
        #步骤：对于某个样本，分为RGB三个通道，图像在每个通道都会有一个二维数组特征图，先把特征图重塑成一维数组，这样每个样本就是有3个一维数组
        #然后和转置相乘，得到3*3的一个R*R、R*G、R*B、G*R等的一个数组，即为Gram矩阵
        style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        self.gram_style = np.array([np.matmul(style_layer_reshape[i,:,:], style_layer_reshape[i,:,:].T) for i in range(style_layer.shape[0])])  #T是转置的意思
        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])  #创建全0矩阵
        for idxn in range(input_layer.shape[0]):
            self.gram_input[idxn, :, :] = np.matmul(self.input_layer_reshape[idxn,:,:],self.input_layer_reshape[idxn,:,:].T) 
        M = input_layer.shape[2]  # input_layer.shape[3] #height*weight
        N = input_layer.shape[1]  # 样本个数
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        style_diff = self.gram_input - self.gram_style 
        #np.sum会计算style_diff 数组中所有元素平方后的总和
        loss = np.sum(style_diff**2) / self.div / style_layer.shape[0] / 4
        return loss
    def backward(self, input_layer, style_layer):
        bottom_diff = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]*input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
             bottom_diff[idxn, :, :] = np.matmul((self.gram_input[idxn,:,:] - self.gram_style[idxn,:,:]).T, self.input_layer_reshape[idxn,:,:]) / self.div / style_layer.shape[0]
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff

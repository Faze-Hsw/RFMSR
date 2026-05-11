"""
像素空间图像分块工具（仿 InvSR 的 ImageSpliterTh）。

用于大图像的分块推理：将大图切成带重叠的 patch，每个 patch 独立走完整
pipeline（VAE encode → 去噪 → VAE decode），然后用高斯加权融合拼回大图。

参考: https://github.com/zsyOAOA/InvSR/blob/master/utils/util_image.py
"""

import math
import numpy as np
import torch


class ImageSpliterTh:
    """像素空间图像分块器（Torch 版本）。

    将输入图像按 pch_size/stride 滑窗切块，支持迭代取 patch、
    更新结果、高斯加权融合。

    与 InvSR 的 ImageSpliterTh 保持一致的接口设计。

    Args:
        im: [B, C, H, W] torch tensor，像素空间输入图（已归一化）
        pch_size: 每个 patch 的边长（像素空间）
        stride: 滑窗步长（像素空间），stride < pch_size 产生重叠
        sf: 超分倍数，输出 patch 尺寸 = 输入 patch 尺寸 × sf
        extra_bs: 每次迭代聚合多少个 patch（组 batch 并行推理）
        weight_type: 融合权重类型，'Gaussian' 或 'ones'
    """

    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1, weight_type='Gaussian'):
        assert weight_type in ['Gaussian', 'ones']
        self.weight_type = weight_type
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width = im.shape
        self.true_bs = bs

        # 计算所有 patch 的起始坐标
        self.height_starts_list = self._extract_starts(height)
        self.width_starts_list = self._extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = len(self.starts_list)
        self.count_pchs = 0

        self.im_ori = im
        self.dtype = torch.float64
        # 输出结果和权重累加图（在超分后的像素空间）
        self.im_res = torch.zeros(
            [bs, chn, height * sf, width * sf],
            dtype=self.dtype, device=im.device,
        )
        self.pixel_count = torch.zeros(
            [bs, chn, height * sf, width * sf],
            dtype=self.dtype, device=im.device,
        )

    def _extract_starts(self, length):
        """计算滑窗起始坐标列表，确保最后一块不超出边界。"""
        if length <= self.pch_size:
            return [0]
        starts = list(range(0, length, self.stride))
        # 确保最后一块回退到 length - pch_size，不超边界
        for ii in range(len(starts)):
            if starts[ii] + self.pch_size > length:
                starts[ii] = length - self.pch_size
        # 去重但保持顺序
        starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return self.length

    def __iter__(self):
        self.count_pchs = 0
        return self

    def __next__(self):
        if self.count_pchs >= self.length:
            raise StopIteration()

        index_infos = []
        current_starts_list = self.starts_list[self.count_pchs:self.count_pchs + self.extra_bs]
        for ii, (h_start, w_start) in enumerate(current_starts_list):
            w_end = w_start + self.pch_size
            h_end = h_start + self.pch_size
            current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
            if ii == 0:
                pch = current_pch
            else:
                pch = torch.cat([pch, current_pch], dim=0)

            # 记录超分后的坐标（用于 update 时写入正确位置）
            index_infos.append([
                h_start * self.sf, h_end * self.sf,
                w_start * self.sf, w_end * self.sf,
            ])

        self.count_pchs += len(current_starts_list)
        return pch, index_infos

    def update(self, pch_res, index_infos):
        """将处理后的 patch 结果写入累加图。

        Args:
            pch_res: [B*extra_bs, C, pch_h*sf, pch_w*sf] 超分后的 patch 结果
            index_infos: [(h_start, h_end, w_start, w_end), ...] 超分后坐标
        """
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii].to(dtype=self.dtype)
            current_weight = self._get_weight(current_pch.shape[-2], current_pch.shape[-1])
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch * current_weight
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += current_weight

    def gather(self):
        """融合所有 patch 结果，返回最终大图。"""
        assert torch.all(self.pixel_count != 0), "存在未覆盖的像素区域！"
        return self.im_res.div(self.pixel_count)

    @staticmethod
    def _generate_kernel_1d(ksize):
        """生成 1D 高斯核（使用 OpenCV 默认 sigma 公式）。"""
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        # 手动计算高斯核（不依赖 cv2）
        if ksize % 2 == 0:
            # 偶数尺寸：先算 ksize+1 再截取，与 InvSR 保持一致
            x = np.arange(ksize + 1) - (ksize) / 2.0
            kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
            kernel = kernel[1:]  # 去掉第一个元素
        else:
            x = np.arange(ksize) - (ksize - 1) / 2.0
            kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.reshape(-1, 1)

    def _get_weight(self, height, width):
        """生成 2D 融合权重矩阵。"""
        if self.weight_type == 'ones':
            kernel = torch.ones(1, 1, height, width)
        elif self.weight_type == 'Gaussian':
            kernel_h = self._generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self._generate_kernel_1d(width).reshape(1, -1)
            kernel = np.matmul(kernel_h, kernel_w)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")
        return kernel.to(dtype=self.dtype, device=self.im_ori.device)

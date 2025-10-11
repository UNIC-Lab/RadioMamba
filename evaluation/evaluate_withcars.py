import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import defaultdict
import math
import glob  # 用于更灵活地查找文件

# 导入 torchmetrics 函数
try:
    from torchmetrics.functional import (
        structural_similarity_index_measure as functional_ssim,
        peak_signal_noise_ratio as functional_psnr
    )
except ImportError:
    functional_ssim = None
    functional_psnr = None
    print("Error: torchmetrics library not found. Please install it using 'pip install torchmetrics'.")
    print("SSIM and PSNR calculations will be skipped.")

# --- 配置参数 ---
# 真实的Ground Truth图片根路径
# 注意：这个路径下应包含所有真实的图片，且文件名与预测图片的文件名相对应
# GT_DATA_ROOT = '/mnt/mydisk/hgjia/data/RadioMapSeer/gain/DPM'
GT_DATA_ROOT = '/mnt/mydisk/hgjia/data/RadioMapSeer/gain/carsDPM'

# 各模型预测结果的图片文件夹路径及其对应的模型名称
# 这些路径下应包含 4000 张预测图片，且图片文件名与 GT_DATA_ROOT 中的文件名一致

thresh_hold = 0.0

PREDICTION_MODEL_DIRS = {
    # 'radiounet': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiounet',
    # 'radiomamba_v9_s1': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v9/stage1',
    # 'radiomamba_v9_s2': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v9/stage2',
    # 'rme_gan': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_rme_gan',
    # 'radiomamba_v10': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v10',
    # 'radiomamba_v4': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v4',
    # 'radiodiff_unic_without_aft': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiodiff_unic_without_aft',
    # 'radiodiff_unic_with_aft': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiodiff_unic_with_aft',
    # 'radiomamba_v11': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v11',
    # 'radiomamba_v12_small_30000steps': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v12/sample_radiomamba_v12_small_30000steps',
    # 'radiomamba_v12_small_39600steps': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v12/sample_radiomamba_v12_small_39600steps',
    # 'radiomamba_v12_medium': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v12/sample_radiomamba_v12_medium'
    # 'radiomamba_v13_nocars': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v13_nocars'
    # 'radiomamba_v13_withcars': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v13_withcars'
    # 'radiomamba_v14_nocars': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v14_nocars'
    'radiomamba_v14_withcars': '/mnt/mydisk/hgjia/scr/RadioMambaUnet/union_compare/sample_radiomamba_v14_withcars'
}

# 你希望评估的图片数量（通常为预测集中的图片数量）
NUM_IMAGES_TO_EVALUATE = 8000

# 结果保存目录
RESULTS_OUTPUT_DIR = '../results/metrics'

# --- 设备设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 图像预处理变换 ---
# 转换为灰度图，然后转换为Tensor，并归一化到 0-1
# 对于多数图像处理，PIL Image的'L'模式直接对应单通道灰度图
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保输出是单通道
    transforms.ToTensor(),  # 自动将PIL Image (0-255) 转换为 Tensor (0-1)
])

# --- 自定义预测图片加载器 (用于加载已经保存的预测图片) ---


class PredictedImageDataset(Dataset):
    def __init__(self, pred_dir, transform=None):
        self.pred_dir = pred_dir
        self.transform = transform if transform else image_transform

        # 获取所有图片路径和文件名
        self.image_paths = []
        self.image_names = []  # 用于匹配GT，存储不带路径的文件名

        # 遍历指定目录下的所有图片文件
        # glob.glob 可以更方便地查找文件
        img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        for ext in img_extensions:
            # os.path.join(pred_dir, ext) 构造路径，例如 '/path/to/preds/*.png'
            for img_path in sorted(glob.glob(os.path.join(pred_dir, ext))):
                self.image_paths.append(img_path)
                self.image_names.append(os.path.basename(img_path))  # 只保留文件名

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files found in the prediction directory: {pred_dir}. Please check the path and image extensions.")

        print(f"Found {len(self.image_paths)} predicted images in {pred_dir}.")

    def __len__(self):
        # 返回找到的图片数量，以便 DataLoader 知道有多少项
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]  # 获取文件名，用于匹配GT

        image = Image.open(img_path).convert('L')  # 确保以灰度模式打开
        image_tensor = self.transform(image)  # 转换为 Tensor，并归一化到 0-1

        return image_tensor, img_name

# --- 指标计算函数 ---


def calculate_metrics_for_pair(pred_tensor, target_tensor):
    """
    计算给定预测张量和目标张量的MSE, NMSE, RMSE, SSIM, PSNR。
    pred_tensor 和 target_tensor 预计都是归一化到 0-1 的 float Tensor。
    形状应为 (B, C, H, W)，其中 C=1。
    """
    metrics = {}

    # Ensure tensors are float32 for metric calculations
    # And add batch/channel dims if they are missing (e.g., (H, W) or (C, H, W))
    pred_tensor = pred_tensor.float()
    target_tensor = target_tensor.float()

    # Ensure tensors have shape (B, 1, H, W) for metric calculations
    if pred_tensor.ndim == 2:  # (H, W) -> (1, 1, H, W)
        pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)
    elif pred_tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W) -> (1, 1, H, W)
        pred_tensor = pred_tensor.unsqueeze(0)
    if pred_tensor.shape[1] > 1:  # If C > 1, take first channel as it's grayscale
        pred_tensor = pred_tensor[:, :1, :, :]

    if target_tensor.ndim == 2:  # (H, W) -> (1, 1, H, W)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif target_tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W) -> (1, 1, H, W)
        target_tensor = target_tensor.unsqueeze(0)
    if target_tensor.shape[1] > 1:  # If C > 1, take first channel as it's grayscale
        target_tensor = target_tensor[:, :1, :, :]

    # MSE (Mean Squared Error)
    mse_loss_fn = nn.MSELoss(reduction='mean')
    mse = mse_loss_fn(pred_tensor, target_tensor).item()
    metrics['MSE'] = mse

    # RMSE (Root Mean Squared Error)
    rmse = math.sqrt(mse)
    metrics['RMSE'] = rmse

    # NMSE (Normalized Mean Squared Error)
    # Defined as sum((pred-target)^2) / sum(target^2)
    # If using MSELoss(reduction='mean'), then it's mean((pred-target)^2) / mean(target^2)
    target_squared_mean = mse_loss_fn(
        target_tensor, torch.zeros_like(target_tensor)).item()
    if target_squared_mean < 1e-9:  # 避免除以零，如果目标值接近零
        if mse < 1e-9:  # 预测值也接近零，则认为是完美匹配
            nmse = 0.0
        else:  # 目标值接近零，但预测值不接近零，说明误差相对于目标值无限大
            nmse = float('inf')
    else:
        nmse = mse / target_squared_mean
    metrics['NMSE'] = nmse

    # SSIM (Structural Similarity Index Measure)
    if functional_ssim:
        # torchmetrics期望 (N, C, H, W)，我们的 pred_tensor/target_tensor 已经处理为 (B, 1, H, W)
        ssim_val = functional_ssim(
            pred_tensor, target_tensor, data_range=1.0).item()
        metrics['SSIM'] = ssim_val
    else:
        metrics['SSIM'] = float('nan')  # 如果torchmetrics未安装，则为NaN

    # PSNR (Peak Signal-to-Noise Ratio)
    if functional_psnr:
        psnr_val = functional_psnr(
            pred_tensor, target_tensor, data_range=1.0).item()
        metrics['PSNR'] = psnr_val
    else:
        metrics['PSNR'] = float('nan')  # 如果torchmetrics未安装，则为NaN

    return metrics

# --- 主评估逻辑 ---


def main():
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    # 1. 预加载 Ground Truth 图片路径映射
    # 遍历GT_DATA_ROOT及其所有子目录，构建文件名到完整路径的映射
    print(f"Scanning Ground Truth directory: {GT_DATA_ROOT}...")
    gt_image_map = {}
    total_gt_images_found = 0
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # 支持的图片格式

    for root, _, files in os.walk(GT_DATA_ROOT):
        for f in files:
            if f.lower().endswith(img_extensions):
                gt_image_map[f] = os.path.join(root, f)
                total_gt_images_found += 1
    print(f"Found {total_gt_images_found} Ground Truth images in total.")

    # 存储每个模型的平均指标，用于最终摘要
    all_models_avg_metrics = defaultdict(dict)

    # 遍历每个模型进行评估
    for model_name, pred_dir_path in PREDICTION_MODEL_DIRS.items():
        print(f"\n--- Evaluating Model: {model_name} ---")
        if not os.path.exists(pred_dir_path):
            print(
                f"Error: Prediction directory for '{model_name}' not found at '{pred_dir_path}'. Skipping.")
            continue

        try:
            # 为当前模型创建一个预测数据集和数据加载器
            pred_dataset = PredictedImageDataset(
                pred_dir_path, transform=image_transform)
            # 限制评估数量为NUM_IMAGES_TO_EVALUATE或实际找到的图片数量
            # min(NUM_IMAGES_TO_EVALUATE, len(pred_dataset))
            # 实际上，如果预测集保证4000张，直接用 len(pred_dataset) 即可，因为我们要遍历所有预测图
            actual_num_images_to_process = min(
                NUM_IMAGES_TO_EVALUATE, len(pred_dataset))

            # DataLoader for predicted images (batch_size=1 is crucial for file-based matching)
            pred_dataloader = DataLoader(
                pred_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)  # num_workers 可根据系统性能调整

            current_model_total_metrics = defaultdict(float)
            processed_count = 0
            skipped_count = 0

            # 使用 tqdm 包装数据加载器，显示进度
            pbar = tqdm(pred_dataloader, total=actual_num_images_to_process,
                        desc=f"Processing {model_name}")

            for pred_image_tensor, pred_image_name_list in pbar:
                if processed_count >= NUM_IMAGES_TO_EVALUATE:  # 达到指定评估数量即停止
                    break

                # pred_image_name_list 是一个列表，因为 batch_size=1，所以取第一个元素
                pred_image_name = pred_image_name_list[0]

                # 查找对应的 Ground Truth 图片路径
                gt_image_path = gt_image_map.get(pred_image_name)

                if gt_image_path is None:
                    # print(f"Warning: Ground Truth for '{pred_image_name}' not found. Skipping.")
                    skipped_count += 1
                    continue

                try:
                    # 加载并预处理 Ground Truth 图片
                    gt_image = Image.open(gt_image_path).convert('L')
                    target_tensor = image_transform(gt_image).to(device)
                    
                    if thresh_hold > 0:
                        mask = target_tensor < thresh_hold
                        target_tensor[mask] = thresh_hold
                        target_tensor = target_tensor - thresh_hold * torch.ones_like(target_tensor)
                        target_tensor = target_tensor / (1 - thresh_hold)
                    

                    # 将预测图片张量移到设备上
                    pred_tensor = pred_image_tensor.to(device)
                    
                    # 计算当前图片对的指标
                    # 注意：calculate_metrics_for_pair 会处理张量形状和通道
                    batch_metrics = calculate_metrics_for_pair(
                        pred_tensor, target_tensor)

                    for k, v in batch_metrics.items():
                        current_model_total_metrics[k] += v  # 累加每个指标的值

                    processed_count += 1

                    # 更新进度条的显示信息（显示当前平均值）
                    pbar.set_postfix({
                        'MSE': f"{current_model_total_metrics['MSE']/processed_count:.4f}",
                        'PSNR': f"{current_model_total_metrics['PSNR']/processed_count:.2f}",
                        'SSIM': f"{current_model_total_metrics['SSIM']/processed_count:.4f}"
                    })

                except Exception as e:
                    print(
                        f"Error processing image '{pred_image_name}' from GT path '{gt_image_path}': {e}. Skipping.")
                    skipped_count += 1
                    continue

            # 计算并保存当前模型的平均指标
            if processed_count > 0:
                print(
                    f"\n--- Average Metrics for {model_name} (over {processed_count} images) ---")

                # 动态生成结果文件名，例如 "radiounet_metric_resu.txt"
                # 将模型名称转换为小写并替换空格为下划线
                output_filename = f"{model_name.lower().replace(' ', '_')}_metric_resu.txt"
                report_file_path = os.path.join(
                    RESULTS_OUTPUT_DIR, output_filename)

                # 使用 'w' 模式打开文件，每次运行都会创建新文件或覆盖旧文件
                with open(report_file_path, 'w') as f:
                    f.write(
                        f"--- Model: {model_name} ({processed_count} images evaluated) ---\n")
                    if skipped_count > 0:
                        f.write(
                            f"({skipped_count} images skipped due to missing GT or errors)\n")

                    for k, v in current_model_total_metrics.items():
                        avg_value = v / processed_count
                        print(f"{k}: {avg_value:.6f}")
                        f.write(f"{k}: {avg_value:.6f}\n")
                        all_models_avg_metrics[model_name][k] = avg_value
            else:
                print(
                    f"No images processed for {model_name}. Skipped {skipped_count} images.")

        except FileNotFoundError as e:
            print(f"Skipping model {model_name} due to: {e}")
            continue
        except Exception as e:
            print(
                f"An unexpected error occurred while processing model {model_name}: {e}")
            continue

    print(f"\nAll evaluation reports saved to: {RESULTS_OUTPUT_DIR}")
    print("\n--- Summary of All Models' Average Metrics ---")
    for model_name, metrics in all_models_avg_metrics.items():
        print(f"Model: {model_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()

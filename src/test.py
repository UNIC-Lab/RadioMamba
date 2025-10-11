# @Description: Testing script for RadioMambaNet v14. Simplified version with single model configuration.

import os
import argparse
import yaml
import glob
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import time

# Import local modules
from train import RadioMapSeerDataModule, LightningRadioModel


def save_prediction_image(pred_tensor, save_path):
    pred_np = pred_tensor.squeeze().cpu().numpy()
    if pred_np.dtype != np.uint8:
        pred_np = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pred_np, mode='L').save(save_path)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir, filename_pattern):
    candidate_checkpoints = glob.glob(os.path.join(
        checkpoint_dir, filename_pattern + "*.ckpt"))
    if not candidate_checkpoints:
        return None

    checkpoints_with_metric = []
    for ckpt in candidate_checkpoints:
        try:
            loss_part = ckpt.split('val_total_loss=')[-1].split('.ckpt')[0]
            metric_val = float(loss_part)
            checkpoints_with_metric.append((metric_val, ckpt))
        except (ValueError, IndexError):
            continue  # Skip files that don't match the loss pattern

    if checkpoints_with_metric:
        checkpoints_with_metric.sort(key=lambda x: x[0])
        return checkpoints_with_metric[0][1]
    else:
        # Fallback to latest modified if no loss value in name
        candidate_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return candidate_checkpoints[0]


def test_model(config_path, specified_checkpoint_path=None, device=None):
    cfg = load_config(config_path)
    testing_cfg = cfg['testing']

    # Build model parameters directly from config (no more multi-scale)
    model_params = {
        'in_channels': cfg['Model']['in_channels'],
        'out_channels': cfg['Model']['out_channels'],
        'dims': cfg['Model']['dims'],
        'depths': cfg['Model']['depths'],
        'ssm_d_state': cfg['Model']['ssm_d_state'],
        'ssm_d_conv': cfg['Model']['ssm_d_conv'],
        'ssm_expand': cfg['Model']['ssm_expand']
    }

    # Use training config directly
    training_config = cfg['training'].copy()

    output_dir = testing_cfg['results_save_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Prediction images will be saved to: {output_dir}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = specified_checkpoint_path or testing_cfg.get(
        'checkpoint_path')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint path '{checkpoint_path}' not found. Attempting to find best checkpoint automatically.")
        checkpoint_dir = cfg['callbacks']['checkpoint_best']['dirpath']
        filename_base = cfg['callbacks']['checkpoint_best']['filename'].split('{')[
            0]
        checkpoint_path = find_best_checkpoint(checkpoint_dir, filename_base)
        if not checkpoint_path:
            raise FileNotFoundError(
                f"Could not find any checkpoint in {checkpoint_dir}. Please specify a valid path in config.")

    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Load model with correct parameters from v14 config
    model = LightningRadioModel.load_from_checkpoint(
        checkpoint_path,
        model_params=model_params,
        training_config=training_config
    )
    model.to(device)
    model.eval()

    # Prepare test data config
    test_data_config = cfg['data'].copy()
    test_data_config['batch_size'] = testing_cfg['test_batch_size']

    data_module = RadioMapSeerDataModule(
        data_config=test_data_config, seed=cfg.get('seed', 42))
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    num_save_images = testing_cfg.get('num_save_images', 4000)
    saved_count = 0
    total_processing_time = 0.0  # 总处理时间
    total_inference_time = 0.0  # 纯推理时间

    print(f"Test dataloader length: {len(test_dataloader)}")
    print(f"Test dataset length: {len(data_module.test_dataset)}")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating Predictions"):
            if saved_count >= num_save_images:
                break

            # 开始计算总处理时间
            batch_start_time = time.time()

            # 现在batch_size=1，所以每个batch只有一张图片
            inputs, _, names = batch
            inputs = inputs.to(device)

            # 单独计算推理时间
            inference_start_time = time.time()
            predictions = model(inputs)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            total_inference_time += inference_time

            # 由于batch_size=1，直接处理第一张（也是唯一一张）图片
            image_name = names[0]
            save_full_path = os.path.join(output_dir, image_name)
            save_prediction_image(predictions[0], save_full_path)
            saved_count += 1

            # 结束计算总处理时间
            batch_end_time = time.time()
            batch_processing_time = batch_end_time - batch_start_time
            total_processing_time += batch_processing_time

            # 每500张图片打印一次进度
            if saved_count % 500 == 0:
                print(
                    f"Processed {saved_count} images, current file: {image_name}")

    # 计算平均时间
    if saved_count > 0:
        avg_processing_time = total_processing_time / saved_count
        avg_inference_time = total_inference_time / saved_count
        avg_io_time = avg_processing_time - avg_inference_time
    else:
        avg_processing_time = avg_inference_time = avg_io_time = 0.0

    print(f"\n--- Prediction Complete ---")
    print(f"Saved {saved_count} prediction images to {output_dir}")
    print(f"Total processing time: {total_processing_time:.8f} seconds")
    print(f"Total inference time: {total_inference_time:.8f} seconds")
    print(f"Average total processing time per image: {avg_processing_time:.8f} seconds")
    print(f"Average inference time per image: {avg_inference_time:.8f} seconds")
    print(f"Average I/O time per image: {avg_io_time:.8f} seconds")

    # 保存详细的平均时间到文件
    time_file_path = f"/mnt/mydisk/hgjia/scr/RadioMambaUnet/average_time_v14_nocars.txt"
    with open(time_file_path, 'w') as f:
        f.write(f"# Detailed timing information per image (seconds)\n")
        f.write(f"Total images processed: {saved_count}\n")
        f.write(f"Model version: v14\n")
        f.write(f"Total processing time: {total_processing_time:.8f}\n")
        f.write(f"Total inference time: {total_inference_time:.8f}\n")
        f.write(f"Average total processing time: {avg_processing_time:.8f}\n")
        f.write(f"Average inference time: {avg_inference_time:.8f}\n")
        f.write(f"Average I/O time: {avg_io_time:.8f}\n")

    print(f"Detailed timing information saved to: {time_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test RadioMambaNet v14 model.")
    parser.add_argument('--config', type=str, default='../configs/config_withcars.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='(Optional) Path to a specific model checkpoint file.')
    args = parser.parse_args()

    # 设置测试设备为 cuda:0
    device = torch.device("cuda:3")
    test_model(args.config, args.checkpoint, device)

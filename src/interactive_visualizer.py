#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description: Interactive Radio Map Prediction Visualizer for RadioMambaNet v14
             Click on building map to place TX antenna and get real-time prediction
"""

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import time

# Import local modules
from train import LightningRadioModel

# ==================== 配置参数 ====================
CHECKPOINT_PATH = '/mnt/mydisk/hgjia/resu_mamba/resu_mamba_v14_nocars/best-radiomamba-v14-nocars-step=26180-val_total_loss=0.0125.ckpt'
BUILDINGS_DIR = '/mnt/mydisk/hgjia/data/RadioMapSeer/png/buildings_complete'
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 预定义 10 个建筑地图（从测试集中选择）
PREDEFINED_MAPS = [
    "289.png",  # 测试集地图
    "419.png",
    "345.png",
    "512.png",
    "678.png",
    "601.png",
    "650.png",
    "555.png",
    "620.png",
    "690.png"
]

# 模型配置参数（与训练时一致）
MODEL_PARAMS = {
    'in_channels': 3,
    'out_channels': 1,
    'dims': [48, 96, 192, 384],
    'depths': [2, 3, 4, 2],
    'ssm_d_state': 32,
    'ssm_d_conv': 4,
    'ssm_expand': 2
}

TRAINING_CONFIG = {
    'learning_rate': 0.0009,
    'weight_decay': 0.0001,
    'loss_weights': {'l1': 0.4, 'mse': 0.1, 'ssim': 0.2, 'gradient': 0.3},
    'lr_scheduler_patience': 8
}

# ==================== 全局变量 ====================
model = None
current_building_map = None

# ==================== 模型加载 ====================
def load_model():
    """加载训练好的模型"""
    global model
    print(f"Loading model from: {CHECKPOINT_PATH}")
    print(f"Using device: {DEVICE}")
    
    model = LightningRadioModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        model_params=MODEL_PARAMS,
        training_config=TRAINING_CONFIG
    )
    model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully!")

# ==================== TX 天线图生成 ====================
def generate_tx_map(click_x, click_y, width=256, height=256):
    """
    在点击位置生成 TX 天线图（模拟原始数据）
    原始数据中 TX 图只有一个像素为 1.0，其余全为 0
    
    Args:
        click_x: 点击的 x 坐标
        click_y: 点击的 y 坐标
        width: 图像宽度
        height: 图像高度
    
    Returns:
        tx_map: numpy array (height, width), 值在 [0, 1]
    """
    tx_map = np.zeros((height, width), dtype=np.float32)
    
    # 只在点击位置设置单个像素为 1.0（与原始数据格式一致）
    if 0 <= click_x < width and 0 <= click_y < height:
        tx_map[click_y, click_x] = 1.0
    
    return tx_map

# ==================== 主推理函数 ====================
def predict_path_loss(evt: gr.SelectData):
    """
    核心推理函数：接收点击事件，生成预测
    
    Args:
        evt: Gradio 的 SelectData 事件，包含点击坐标
    
    Returns:
        building_with_marker: 标记了点击位置的建筑图
        prediction_viz: 预测的路径损耗可视化图
        coord_info: 坐标信息文本
        time_info: 推理时间信息
    """
    global current_building_map
    
    if model is None:
        return None, None, "❌ 模型未加载", ""
    
    if current_building_map is None:
        return None, None, "❌ 请先选择建筑地图", ""
    
    # 获取点击坐标
    # Gradio evt.index 格式: [x, y] (即 [column, row])
    # 但 numpy 数组索引格式: [row, column] (即 [y, x])
    click_x, click_y = evt.index[0], evt.index[1]
    
    # 打印调试信息
    print(f"[DEBUG] Gradio evt.index: {evt.index}, click_x={click_x}, click_y={click_y}")
    
    # 坐标信息
    coord_text = f"📍 点击坐标: X={click_x}, Y={click_y}, Z=1.5m (地图: {current_building_map})"
    
    # 加载建筑地图
    building_path = os.path.join(BUILDINGS_DIR, current_building_map)
    if not os.path.exists(building_path):
        return None, None, f"❌ 建筑地图不存在: {building_path}", ""
    
    building_img = Image.open(building_path).convert('L')
    building_np = np.array(building_img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
    
    # 生成 TX 天线图（只有一个像素为 1.0，与原始训练数据格式一致）
    # 注意：generate_tx_map 内部使用 [y, x] 索引
    tx_map = generate_tx_map(click_x, click_y, width=256, height=256)
    
    # 准备 3 通道输入: (建筑, TX, 建筑)
    # 添加 channel 维度并拼接
    building_channel = np.expand_dims(building_np, axis=0)  # (1, H, W)
    tx_channel = np.expand_dims(tx_map, axis=0)  # (1, H, W)
    
    # 拼接为 (3, H, W)
    input_tensor = np.concatenate([building_channel, tx_channel, building_channel], axis=0)
    
    # 转换为 PyTorch 张量并添加 batch 维度
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(DEVICE)  # (1, 3, H, W)
    
    # 模型推理
    start_time = time.time()
    with torch.no_grad():
        prediction = model(input_tensor)
    inference_time = time.time() - start_time
    
    # 后处理：clamp 并转为 numpy
    prediction_np = torch.clamp(prediction[0, 0], 0.0, 1.0).cpu().numpy()  # (H, W)
    
    # 可视化1: 在建筑图上标记点击位置
    building_rgb = np.stack([building_np]*3, axis=-1)  # 转为 RGB (H, W, 3)
    # 画一个红色十字标记
    # 注意：numpy 索引是 [row, col] = [y, x]
    marker_size = 5
    # 垂直线（固定 x，变化 y）
    y_start = max(0, click_y - marker_size)
    y_end = min(256, click_y + marker_size + 1)
    if 0 <= click_x < 256:
        building_rgb[y_start:y_end, click_x, :] = [1.0, 0.0, 0.0]
    
    # 水平线（固定 y，变化 x）
    x_start = max(0, click_x - marker_size)
    x_end = min(256, click_x + marker_size + 1)
    if 0 <= click_y < 256:
        building_rgb[click_y, x_start:x_end, :] = [1.0, 0.0, 0.0]
    
    # 可视化2: 预测结果使用灰度图（参考 test.py）
    # 直接将归一化的预测值转换为 uint8 灰度图，并转为 RGB 格式供 Gradio 显示
    prediction_gray = (np.clip(prediction_np, 0, 1) * 255).astype(np.uint8)
    prediction_viz = np.stack([prediction_gray]*3, axis=-1)  # 转为 RGB 格式
    
    # 转换为 uint8 格式供 Gradio 显示
    building_viz = (building_rgb * 255).astype(np.uint8)
    
    # 时间信息
    time_text = f"⚡ 推理时间: {inference_time:.4f} 秒"
    
    return building_viz, prediction_viz, coord_text, time_text

# ==================== 地图选择函数 ====================
def load_building_map(building_map_name):
    """
    加载选中的建筑地图
    
    Args:
        building_map_name: 建筑地图文件名
    
    Returns:
        building_img: PIL Image
    """
    global current_building_map
    
    # 处理 None 的情况（可能来自手动输入后的更新）
    if building_map_name is None:
        if current_building_map is not None:
            building_map_name = current_building_map
        else:
            return np.zeros((256, 256, 3), dtype=np.uint8)
    
    building_path = os.path.join(BUILDINGS_DIR, building_map_name)
    
    if not os.path.exists(building_path):
        # 如果文件不存在，返回空图
        return np.zeros((256, 256, 3), dtype=np.uint8)
    
    building_img = Image.open(building_path).convert('RGB')
    current_building_map = building_map_name
    
    return building_img


def load_manual_map(map_number):
    """
    根据手动输入的编号加载建筑地图
    
    Args:
        map_number: 建筑地图编号（0-700）
    
    Returns:
        building_img: PIL Image
        status_msg: 状态消息
    """
    global current_building_map
    
    try:
        map_num = int(map_number)
        if map_num < 0 or map_num > 700:
            return np.zeros((256, 256, 3), dtype=np.uint8), f"❌ 编号必须在 0-700 之间"
        
        building_map_name = f"{map_num}.png"
        building_path = os.path.join(BUILDINGS_DIR, building_map_name)
        
        if not os.path.exists(building_path):
            return np.zeros((256, 256, 3), dtype=np.uint8), f"❌ 地图文件不存在: {building_map_name}"
        
        building_img = Image.open(building_path).convert('RGB')
        current_building_map = building_map_name
        
        return building_img, f"✓ 成功加载地图: {building_map_name}"
    
    except ValueError:
        return np.zeros((256, 256, 3), dtype=np.uint8), "❌ 请输入有效的数字"

# ==================== Gradio 界面 ====================
def create_interface():
    """创建 Gradio 交互界面"""
    
    with gr.Blocks(title="RadioMamba 交互式可视化工具") as demo:
        gr.Markdown("""
        # 🎯 RadioMamba 实时路径损耗预测工具
        
        **使用说明：**
        1. 从下拉框选择预定义的建筑地图，或者手动输入地图编号（0-700）
        2. 在左侧建筑图上点击任意位置放置发射机（TX）
        3. 右侧自动显示预测的路径损耗分布图
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                map_selector = gr.Dropdown(
                    choices=PREDEFINED_MAPS,
                    value=PREDEFINED_MAPS[0],
                    label="📁 方式1：从下拉框选择建筑地图",
                    interactive=True
                )
            with gr.Column(scale=1):
                map_number_input = gr.Textbox(
                    label="🔢 方式2：手动输入地图编号（0-700）",
                    placeholder="例如: 289",
                    interactive=True
                )
                load_manual_btn = gr.Button("确定", variant="primary")
        
        with gr.Row():
            manual_status = gr.Textbox(
                label="📋 加载状态",
                value="",
                interactive=False
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📍 输入：建筑地图（点击选择TX位置）")
                building_display = gr.Image(
                    type="numpy",
                    label="Building Map",
                    interactive=True,
                    height=256  # 与原始图像尺寸一致，避免坐标缩放偏差
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📡 输出：预测路径损耗")
                prediction_display = gr.Image(
                    type="numpy",
                    label="Predicted Path Loss",
                    interactive=False,
                    height=256  # 保持与输入一致的显示尺寸
                )
        
        with gr.Row():
            coord_info = gr.Textbox(
                label="📍 坐标信息",
                value="请在左侧地图上点击选择 TX 位置",
                interactive=False
            )
            time_info = gr.Textbox(
                label="⏱️ 性能统计",
                value="",
                interactive=False
            )
        
        # 事件绑定
        # 1. 下拉框地图选择事件
        map_selector.change(
            fn=load_building_map,
            inputs=[map_selector],
            outputs=[building_display]
        )
        
        # 2. 手动输入地图编号事件
        load_manual_btn.click(
            fn=load_manual_map,
            inputs=[map_number_input],
            outputs=[building_display, manual_status]
        )
        
        # 3. 点击事件（核心功能）
        building_display.select(
            fn=predict_path_loss,
            inputs=[],
            outputs=[building_display, prediction_display, coord_info, time_info]
        )
        
        # 初始化：加载第一张地图
        demo.load(
            fn=load_building_map,
            inputs=[map_selector],
            outputs=[building_display]
        )
    
    return demo

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("🚀 RadioMamba 交互式可视化工具启动中...")
    print("="*60)
    
    # 加载模型
    load_model()
    
    # 创建并启动界面
    demo = create_interface()
    
    print("\n" + "="*60)
    print("✓ 界面已启动！")
    print("📱 访问地址将在下方显示...")
    print("="*60 + "\n")
    
    # 启动服务器
    demo.launch(
        server_name="0.0.0.0",  # 允许远程访问
        server_port=7860,
        share=False,  # 如需公网访问可设为 True
        inbrowser=False  # 服务器环境不自动打开浏览器
    )


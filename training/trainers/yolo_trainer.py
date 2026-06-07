from ultralytics import YOLO
import os
from pathlib import Path

class CustomYoloTrainer:
    """
    面向对象的YOLOv8训练器类。
    集成混合精度训练(AMP)、早停机制(Early Stopping)以及实验结果重定向。
    基于面向对象思想重构训练器。
    """
    def __init__(self, model_variant='yolov8s.pt', project_name='mask_detection_exp', experiment_name='baseline'):
        """
        初始化训练器。
        
        Args:
            model_variant (str): YOLOv8模型变体，如 'yolov8n.pt', 'yolov8s.pt' 等。
            project_name (str): 实验项目名称，将作为子目录存在于 experiments/ 下。
            experiment_name (str): 具体实验名称（如 baseline, no_mosaic 等）。
        """
        self.model = YOLO(model_variant)
        self.project_path = Path("experiments") / project_name
        self.experiment_name = experiment_name
        
        # 确保 experiments 目录存在
        self.project_path.parent.mkdir(exist_ok=True)

    def train(self, data_config='configs/dataset.yaml', epochs=100, imgsz=640, batch=16, 
              patience=50, amp=True, mosaic=1.0, mixup=0.0):
        """
        执行模型训练。
        
        Args:
            data_config (str): 数据集配置文件路径。
            epochs (int): 总迭代轮数。
            imgsz (int): 输入图像尺寸。
            batch (int): 批处理大小。
            patience (int): 早停机制的容忍轮数（0表示关闭）。
            amp (bool): 是否开启混合精度训练。
            mosaic (float): Mosaic数据增强概率。
            mixup (float): MixUp数据增强概率。
        """
        print(f"==> Starting Experiment: {self.experiment_name}")
        print(f"==> Model: {self.model.model_name}, AMP: {amp}, Early Stopping Patience: {patience}")
        
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            amp=amp,
            project=str(self.project_path),
            name=self.experiment_name,
            mosaic=mosaic,
            mixup=mixup,
            exist_ok=True,
            # 其他超参数可根据需要在此添加
        )
        return results

    def validate(self):
        """
        在验证集上评估模型表现。
        """
        metrics = self.model.val()
        return metrics

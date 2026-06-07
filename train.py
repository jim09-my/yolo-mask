from training.trainers.yolo_trainer import CustomYoloTrainer
import argparse

def run_experiments():
    # 1. 基础实验 (Baseline): YOLOv8s, 开启混合精度和早停
    baseline_trainer = CustomYoloTrainer(
        model_variant='yolov8s.pt', 
        project_name='face_mask_detection', 
        experiment_name='baseline_s'
    )
    baseline_trainer.train(epochs=10, imgsz=640, batch=16, patience=5) # 示例设为10轮，实际建议100+

    # 2. 消融实验 (Ablation Study): 关闭 Mosaic 增强
    no_mosaic_trainer = CustomYoloTrainer(
        model_variant='yolov8s.pt', 
        project_name='face_mask_detection', 
        experiment_name='ablation_no_mosaic'
    )
    no_mosaic_trainer.train(epochs=10, imgsz=640, batch=16, mosaic=0.0)

    # 3. 多尺度对比 (Scale Comparison): 训练轻量化 YOLOv8n
    nano_trainer = CustomYoloTrainer(
        model_variant='yolov8n.pt', 
        project_name='face_mask_detection', 
        experiment_name='comparison_nano'
    )
    nano_trainer.train(epochs=10, imgsz=640, batch=16)

if __name__ == "__main__":
    # 提示：实际运行前请确保 data/augmented 目录已通过 scripts/augment.py 生成
    # 或者将 configs/dataset.yaml 中的路径指向原始处理后的数据
    run_experiments()

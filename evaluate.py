import argparse
from ultralytics import YOLO
from pathlib import Path
import json
import time

def evaluate_model(model_path, data_config='configs/dataset.yaml', project='experiments', name='evaluation'):
    """
    对训练好的模型进行全面评估。
    计算 mAP, Precision, Recall, 以及推理延迟 (FPS)。
    """
    print(f"==> Loading model for evaluation: {model_path}")
    model = YOLO(model_path)
    
    # 1. 运行验证，获取核心指标
    # split='test' 确保是在测试集上运行
    start_time = time.time()
    results = model.val(data=data_config, split='test', project=project, name=name, exist_ok=True)
    end_time = time.time()
    
    # 2. 提取关键指标
    # mAP@0.5, mAP@0.5:0.95, Precision, Recall
    mAP50 = results.box.map50
    mAP95 = results.box.map
    precision = results.box.mp
    recall = results.box.mr
    
    # 3. 计算推理延迟 (Inference Speed)
    # 注意：model.val() 会返回每张图的平均处理时间 (ms)
    speed_dict = results.speed
    preprocess = speed_dict.get('preprocess', 0)
    inference = speed_dict.get('inference', 0)
    postprocess = speed_dict.get('postprocess', 0)
    total_time_per_img = preprocess + inference + postprocess
    fps = 1000 / total_time_per_img if total_time_per_img > 0 else 0

    print("\n" + "="*30)
    print(f"Evaluation Results for: {model_path}")
    print(f"mAP@0.5: {mAP50:.4f}")
    print(f"mAP@0.5:0.95: {mAP95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Speed: {total_time_per_img:.2f} ms/img (FPS: {fps:.1f})")
    print("="*30 + "\n")
    
    # 4. 保存为 JSON 报告，方便后续绘图对比
    report = {
        "model": model_path,
        "mAP50": float(mAP50),
        "mAP95": float(mAP95),
        "precision": float(precision),
        "recall": float(recall),
        "fps": float(fps),
        "ms_per_img": float(total_time_per_img)
    }
    
    report_path = Path(project) / name / "metrics_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Metrics report saved to: {report_path}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Evaluation Script")
    parser.add_argument("--weights", type=str, default="experiments/face_mask_detection/baseline_s/weights/best.pt", 
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--name", type=str, default="test_set_evaluation", help="Name for evaluation result folder")
    
    args = parser.parse_args()
    
    # 检查权重文件是否存在
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found at {args.weights}")
        print("Please ensure you have trained the model or provided the correct path.")
    else:
        evaluate_model(args.weights, data_config=args.data, name=args.name)

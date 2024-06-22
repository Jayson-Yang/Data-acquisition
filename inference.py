# inference.py
from mmdet.apis import init_detector, inference_detector
import os
import numpy as np

class InferenceEngine:
    def __init__(self):
        config_file = './checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        self.model = init_detector(config_file, checkpoint_file, device='cpu')

    def infer(self, image_path):
        result = inference_detector(self.model, image_path)
        return self.process_results(result, image_path)

    def process_results(self, result, image_path):
        pred_instances = result.pred_instances

        # 提取bbox和标签信息
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        processed_results = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if score >= 0.3:  # 设定阈值以过滤低置信度的框
                left, top, right, bottom = bbox[:4]
                class_name = self.model.dataset_meta['classes'][label]
                processed_results.append({
                    'image_path': image_path,
                    'bbox': [left, top, right, bottom],
                    'class_name': class_name,
                    'score': score
                })
        return processed_results

    def save_results(self, results, output_file='inference_output/inference_results_imagenet.txt'):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a') as f:
            for result in results:
                left, top, right, bottom = result['bbox']
                class_name = result['class_name']
                image_path = result['image_path']
                f.write(f"{image_path} {left:.4f} {top:.4f} {right:.4f} {bottom:.4f} {class_name}\n")

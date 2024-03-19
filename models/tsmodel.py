import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .extractors import build_extractor


class TSModel(nn.Module):
    def __init__(self, extractor, **kwargs):
        super().__init__()
        self.teacher = build_extractor(extractor, pretrained=True)
        self.student = build_extractor(extractor)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def predict(self, image, top_k=0.03):
        b, c, h, w = image.shape
        teacher_outputs = self.teacher(image)
        student_outputs = self.student(image)
        distances = [
            (1 - F.cosine_similarity(teacher_output, student_output)).unsqueeze(1)
            for teacher_output, student_output in zip(teacher_outputs, student_outputs)
        ]

        anomaly_map = torch.zeros((b, 1, h, w), device=image.device)
        for distance in distances:
            distance = F.interpolate(distance, size=image.shape[-2:], mode='bilinear', align_corners=True)
            anomaly_map += distance

        top_k = int(h * w * top_k)
        anomaly_score = np.mean(
            anomaly_map.reshape(b, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
            axis=1)

        return anomaly_score, anomaly_map.cpu().numpy()

    def forward(self, image):
        with torch.no_grad():
            teacher_outputs = self.teacher(image)
        student_outputs = self.student(image)

        outputs = {
            "teacher_outputs": teacher_outputs,
            "student_outputs": student_outputs
        }

        return outputs



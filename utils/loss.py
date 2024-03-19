import torch
import torch.nn as nn
import torch.nn.functional as F


class TSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        loss = 0.
        teacher_outputs = outputs['teacher_outputs']
        student_outputs = outputs['student_outputs']
        for teacher_output, student_output in zip(teacher_outputs, student_outputs):
            distance = 1 - F.cosine_similarity(teacher_output, student_output)
            d_hard = torch.quantile(distance, q=0.95)
            loss += torch.mean(distance[distance >= d_hard])

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, hidden_weight=1.0, logit_weight=0.5, temperature=2.0):
        super().__init__()
        self.hidden_weight = hidden_weight
        self.logit_weight = logit_weight
        self.temperature = temperature

    def forward(self, student_hidden, teacher_hidden, student_logits, teacher_logits):
        # Hidden state MSE loss
        hidden_loss = 0.0
        for s_h, t_h in zip(student_hidden, teacher_hidden):
            hidden_loss += F.mse_loss(s_h, t_h)
        hidden_loss /= len(student_hidden)

        if self.logit_weight == 0:
            return self.hidden_weight * hidden_loss

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        logit_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        logit_loss *= self.temperature**2

        return self.hidden_weight * hidden_loss + self.logit_weight * logit_loss

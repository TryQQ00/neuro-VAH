import torch
import torch.nn as nn

def iv_loss(I_pred, I_ref, V=None, use_log=False, weights=None):
    eps = 1e-12
    if weights is None:
        weights = {}
    lambda_log = weights.get('lambda_log', 0.0)
    lambda_neg = weights.get('lambda_neg', 0.0)
    lambda_mon = weights.get('lambda_mon', 0.0)
    lambda_i0 = weights.get('lambda_i0', 0.0)
    eps = weights.get('eps', eps)

    I_pred = I_pred.squeeze(-1)
    I_ref = I_ref.squeeze(-1)
    loss_mse = nn.MSELoss()(I_pred, I_ref)
    loss = loss_mse

    if use_log and lambda_log > 0:
        log_pred = torch.log(I_pred + eps)
        log_ref = torch.log(I_ref + eps)
        loss_log = nn.MSELoss()(log_pred, log_ref)
        loss = loss + lambda_log * loss_log
    else:
        loss_log = None

    if (not use_log) and (lambda_neg > 0):
        neg_violation = torch.clamp(-I_pred, min=0.0)
        loss_neg = torch.mean(neg_violation ** 2)
        loss = loss + lambda_neg * loss_neg
    else:
        loss_neg = None

    if lambda_mon > 0:
        diff = I_pred[:, 1:] - I_pred[:, :-1]
        mono_violation = torch.clamp(-diff, min=0.0)
        loss_mon = torch.mean(mono_violation ** 2)
        loss = loss + lambda_mon * loss_mon
    else:
        loss_mon = None

    if (not use_log) and (lambda_i0 > 0):
        I0_pred = I_pred[:, 0]
        loss_i0 = torch.mean(I0_pred ** 2)
        loss = loss + lambda_i0 * loss_i0
    else:
        loss_i0 = None

    return loss, {
        'mse': loss_mse, 'log': loss_log,
        'neg': loss_neg, 'mon': loss_mon, 'i0': loss_i0
    }

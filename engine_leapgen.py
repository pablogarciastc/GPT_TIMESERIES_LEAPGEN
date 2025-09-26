# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
# Further modification:
# Modified for MOMENT transformer compatibility
# ------------------------------------------
"""
Train and eval functions used in main.py - Modified for MOMENT compatibility
"""
import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn import MSELoss

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import logging

# for attribute matching of tasks
from attribute_matching import num_new_prompts, get_descriptors_embedding1
from timm.scheduler import create_scheduler

# import clip
import random


# from kan import *

def train_one_epoch(model: torch.nn.Module,
                    criterion, data_loader: Iterable,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    optimizer=None,
                    old_prompt_matcher=None,
                    old_prompt=None,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None,
                    old_num_k=5, ):
    model.train(set_training_mode)

    s = old_num_k

    # Freezing previous tasks' filters
    for name, param in model.named_parameters():
        if name.find('e_prompt.v_conv_vals') >= 0 or name.find('e_prompt.k_conv_vals') >= 0:
            for i in range(s):
                if name.find('.{}.weight'.format(i)) >= 0 or name.find('.{}.bias'.format(i)) >= 0:
                    param.requires_grad = False

    if hasattr(args, 'distributed') and args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")

    # metric_logger.add_meter('Lr_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if hasattr(args, 'SLCA') and args.SLCA:
        metric_logger.add_meter('Lr_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Lr_rps', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # Masking and computing loss
        known_classes = task_id * len(class_mask[0])
        cur_targets = torch.where(target - known_classes >= 0, target - known_classes, -100)
        loss = criterion(logits[:, known_classes:], cur_targets)  # base criterion (CrossEntropyLoss)

        if args.use_e_prompt or args.use_g_prompt:
            if task_id > 0:
                l1_loss = 0.0
                for old_wt, new_wt in zip(old_prompt_matcher.parameters(), model.e_prompt.prompt_embed_matcher.parameters()):
                    l1_loss += torch.norm(old_wt.detach() - new_wt, p=1)
                loss = loss + 0.01 * l1_loss
                prompt_loss = torch.norm(old_prompt.detach() - model.e_prompt.prompt, p=1)
                loss = loss + 0.01 * prompt_loss

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        # Fixed gradient clipping
        if hasattr(args, 'use_clip_grad') and args.use_clip_grad and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        elif max_norm > 0:  # Fallback if use_clip_grad not in args
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())

        if hasattr(args, 'SLCA') and args.SLCA:
            metric_logger.update(Lr_cls=optimizer.param_groups[0]["lr"])
            metric_logger.update(Lr_rps=optimizer.param_groups[1]["lr"])
        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

from timm.utils import accuracy
import utils
from attribute_matching import get_descriptors_embedding1
import logging
import copy


def train_one_epoch_with_aux(
    model, original_model, criterion, data_loader, aux_desc_emb,
    device, epoch, max_norm=0, optimizer=None,
    old_prompt_matcher=None, old_prompt=None,
    set_training_mode=True, task_id=-1, class_mask=None, args=None,
    old_num_k=5,
):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", utils.SmoothedValue(window_size=1, fmt="{value:.5f}"))
    metric_logger.add_meter("Loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("CE_Loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))
    metric_logger.add_meter("Prompt_Loss", utils.SmoothedValue(window_size=1, fmt="{value:.3f}"))

    header = f"Train: Epoch[{epoch + 1}/{args.epochs}]"

    for inp, target in metric_logger.log_every(data_loader, args.print_freq, header):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Get features from original model if available
        cls_features = None
        if original_model is not None:
            with torch.no_grad():
                out0 = original_model(inp)
                cls_features = out0["pre_logits"]

        # Forward pass
        out = model.forwardA1(inp, target, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        lg_any = out['logits']
        logits = lg_any['logits'] if isinstance(lg_any, dict) else lg_any

        # IMPROVED LOSS COMPUTATION
        total_loss = 0.0

        # 1. Current task loss (main objective)
        current_classes = class_mask[task_id]
        current_start = sum(len(class_mask[i]) for i in range(task_id))
        current_end = current_start + len(current_classes)

        # Map global targets to local task targets
        local_targets = target - current_start
        valid_mask = (local_targets >= 0) & (local_targets < len(current_classes))

        if valid_mask.sum() > 0:
            current_logits = logits[:, current_start:current_end]
            ce_loss = criterion(current_logits[valid_mask], local_targets[valid_mask])
            total_loss += ce_loss
            metric_logger.update(CE_Loss=ce_loss.item())

        # 2. Replay loss for previous tasks (prevent catastrophic forgetting)
        if task_id > 0 and args.replay_loss_weight > 0:
            prev_end = current_start
            if prev_end > 0:
                prev_logits = logits[:, :prev_end]
                # Use soft targets from original model
                if original_model is not None:
                    with torch.no_grad():
                        teacher_out = original_model(inp, task_id=-1)  # All tasks
                        teacher_logits = teacher_out['logits']
                        if isinstance(teacher_logits, dict):
                            teacher_logits = teacher_logits['logits']
                        teacher_probs = F.softmax(teacher_logits[:, :prev_end] / args.temperature, dim=1)

                    student_log_probs = F.log_softmax(prev_logits / args.temperature, dim=1)
                    replay_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                    total_loss += args.replay_loss_weight * replay_loss

        # 3. Prompt regularization losses
        prompt_loss = 0.0

        # Pull constraint (encourage prompt diversity)
        if getattr(args, "pull_constraint", False) and out.get("reduce_sim", None) is not None:
            pull_loss = -args.pull_constraint_coeff * out["reduce_sim"]
            prompt_loss += pull_loss

        # Task-specific prompt loss
        if out.get("reduce_sim2", None) is not None:
            task_loss = -args.pull_constraint_coeff2 * out["reduce_sim2"]
            prompt_loss += task_loss

        # L2 regularization on prompts (prevent overfitting)
        if hasattr(model, 'e_prompt') and args.prompt_l2_weight > 0:
            prompt_params = []
            for name, param in model.e_prompt.named_parameters():
                if 'prompt_key' in name or any(gen_name in name for gen_name in ['k_comp_gen', 'v_comp_gen']):
                    prompt_params.append(param)

            if prompt_params:
                l2_loss = sum(torch.norm(p, p=2) for p in prompt_params)
                prompt_loss += args.prompt_l2_weight * l2_loss

        total_loss += prompt_loss
        metric_logger.update(Prompt_Loss=prompt_loss.item())

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Logging
        metric_logger.update(Loss=total_loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(model, data_loader, device, task_id=-1, class_mask=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: [Task {task_id+1}]"
    model.eval()
    correct, total = 0, 0

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input, target = input.to(device), target.to(device)
        out = model(input, task_id=task_id)
        lg_any = out["logits"]
        logits = lg_any["logits"] if isinstance(lg_any, dict) else lg_any

        if args.task_inc and class_mask is not None:
            mask = torch.tensor(class_mask[task_id], dtype=torch.int64, device=device)
            logits_mask = torch.ones_like(logits) * float("-inf")
            logits_mask = logits_mask.index_fill(1, mask, 0.0)
            logits = logits + logits_mask

        loss = criterion(logits, target)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum()
        total += target.size(0)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        metric_logger.meters["Loss"].update(loss.item())
        metric_logger.meters["Acc@1"].update(acc1.item(), n=input.size(0))
        metric_logger.meters["Acc@5"].update(acc5.item(), n=input.size(0))

    metric_logger.synchronize_between_processes()
    print(f"* Acc@1 {metric_logger.meters['Acc@1'].global_avg:.3f} "
          f"Acc@5 {metric_logger.meters['Acc@5'].global_avg:.3f} "
          f"loss {metric_logger.meters['Loss'].global_avg:.3f}")
    return {k: m.global_avg for k, m in metric_logger.meters.items()}, total, correct



@torch.no_grad()
def evaluate_with_aux(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, aux_desc_emb,
                      device, task_id=-1, class_mask=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    correct = 0
    total = 0

    true_labels = []
    pred_labels = []

    # Batchwise Eval time
    start_eval_time = time.time()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):

            # aux = get_nearest_embedding1(input, target, aux_desc_emb)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

            output = model.forwardA1(input, None, -1, cls_features=cls_features, train=False)
            lg_any = output['logits']
            logits = lg_any['logits'] if isinstance(lg_any, dict) else lg_any

            if hasattr(args, 'task_inc') and args.task_inc and class_mask is not None:
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            max_label = torch.max(target).item()
            if logits.shape[1] <= max_label:
                pf = torch.zeros((logits.shape[0], max_label + 1 - logits.shape[1]), device=logits.device)
                logits = torch.cat((logits, pf), dim=1)

            loss = criterion(logits, target)
            # print('Loss')
            predicts = torch.max(logits, dim=1)[1]
            correct += (predicts == target).sum()
            total += len(target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            pred = torch.argmax(logits, dim=1)

            pred_labels.extend(pred.tolist())
            true_labels.extend(target.tolist())

    end_eval_time = time.time()
    # gather the stats from all processesde
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                  losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                 .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                         losses=metric_logger.meters['Loss']))

    print(f"Batchwise eval time for task {task_id + 1} = {(end_eval_time - start_eval_time) / len(data_loader)}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total, correct


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id + 1):
        logging.info('Evaluating task {}...'.format(i + 1))
        test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader[i]['val'],
                                                        device=device, task_id=i, class_mask=class_mask, args=args, )

        # test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader[i]['val'],
        #                     device=device, task_id=i, class_mask=class_mask, args=args,)

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total) * 100.0
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id + 1,
                                                                                                     final_acc,
                                                                                                     avg_stat[1],
                                                                                                     avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    return test_stats


@torch.no_grad()
def evaluate_till_now_with_aux(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, aux_desc_emb,
                               device, task_id=-1, class_mask=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id + 1):
        logging.info('Evaluating task {}...'.format(i + 1))

        test_stats, temp_total, temp_correct = evaluate_with_aux(model=model, original_model=original_model,
                                                                 data_loader=data_loader[i]['val'],
                                                                 aux_desc_emb=aux_desc_emb,
                                                                 device=device, task_id=i, class_mask=class_mask,
                                                                 args=args, )

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total) * 100.0
    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id + 1,
                                                                                                     final_acc,
                                                                                                     avg_stat[1],
                                                                                                     avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)
    print("ACC_MATRIX: ",acc_matrix)

    return test_stats


def train_and_evaluate(model, original_model, criterion, data_loader, lr_scheduler, optimizer, device, class_mask=None,
                       args=None):
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    old_num_k = 0
    aux_desc_emb = None

    for task_id in range(args.num_tasks):
        print(f"\n=== Task {task_id} ===")

        # --- generar embeddings de descriptores
        new_desc = torch.stack(get_descriptors_embedding1(class_mask, task_id, args)).to(device)
        aux_desc_emb = new_desc if task_id == 0 else torch.cat((aux_desc_emb, new_desc), dim=0)

        # --- actualizar la cabeza de clasificación para la nueva tarea
        if task_id > 0:
            model.head.update(len(class_mask[task_id]))

        # --- registrar prompts para nueva tarea EN AMBOS MODELOS
        if args.use_e_prompt:
            curr_num_k = args.top_k_l

            if task_id == 0:
                model.e_prompt.process_new_task(0, curr_num_k, new_desc)
                # SINCRONIZAR original_model
                if original_model is not None:
                    original_model.e_prompt.process_new_task(0, curr_num_k, new_desc)
            else:
                model.e_prompt.process_new_task(old_num_k, old_num_k + curr_num_k, aux_desc_emb)
                # SINCRONIZAR original_model
                if original_model is not None:
                    original_model.e_prompt.process_new_task(old_num_k, old_num_k + curr_num_k, aux_desc_emb)
        else:
            curr_num_k = 0

        # --- reinicializar optimizador
        main_params = [p for n, p in model.named_parameters() if 'prompt_key2' not in n]
        optimizer = optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                            eta_min=1e-6) if args.sched != "constant" else None

        # --- entrenamiento
        for epoch in range(args.epochs):
            train_stats = train_one_epoch_with_aux(
                model=model, original_model=original_model, criterion=criterion,
                data_loader=data_loader[task_id]["train"], aux_desc_emb=aux_desc_emb,
                optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_grad,
                set_training_mode=True, task_id=task_id, class_mask=class_mask,
                args=args, old_num_k=old_num_k,
            )
            if lr_scheduler:
                lr_scheduler.step()

        old_num_k += curr_num_k

        # --- evaluación en todas las tareas vistas hasta ahora
        for i in range(task_id + 1):
            test_stats, total, correct = evaluate(
                model, data_loader[i]["val"],
                device=device, task_id=i,
                class_mask=class_mask, args=args
            )
            acc_matrix[i, task_id] = test_stats["Acc@1"]

        # --- métricas de forgetting/backward transfer
        diag = np.diag(acc_matrix)
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diag)[:task_id])
            print(f"Forgetting={forgetting:.4f}, Backward={backward:.4f}")

    return acc_matrix


def classify_with_proto(data, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    logits = []
    scale_factor = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, len(proto)):
        # class_proto = proto[i].cuda()
        class_proto = proto[i].to(device)
        c_logit = []
        for j in range(class_proto.shape[0]):
            logit = scale_factor * torch.nn.functional.cosine_similarity(
                data,
                class_proto[j],
                dim=-1)
            c_logit.append(logit)
        # c_logit = torch.stack(c_logit).min(dim=0).values
        c_logit = torch.stack(c_logit).mean(dim=0)
        # print("c_logit shape")
        # print(c_logit.shape)

        # print(logit.shape)

        logits.append(c_logit)

    logits = torch.stack(logits).transpose(0, 1)

    return logits


def get_nearest_embedding(data, label, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    embeds = []
    scale_factor = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for i in range(0,len(proto)):
    # batch_proto = proto[label.tolist()]
    batch_proto = [proto[i] for i in label.tolist()]
    for i in range(len(batch_proto)):
        mylist = [*range(0, batch_proto[i].shape[0], 1)]
        idx = random.sample(mylist, 1)
        embeds.append(batch_proto[i][idx, :])

    # embeds = torch.stack(embeds).cuda()
    embeds = torch.stack(embeds).to(device)
    # print(embeds.shape)
    return embeds


def get_nearest_embedding1(data, label, proto):
    # print("Testing with prototype-based cosine cosine similarity")
    embeds = []
    scale_factor = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for i in range(0,len(proto)):
    # batch_proto = proto[label.tolist()]
    batch_proto = [proto[i] for i in label.tolist()]
    # for i in range(len(batch_proto)):
    #     mylist = [*range(0, batch_proto[i].shape[0], 1)]
    #     idx = random.sample(mylist, 1)
    #     embeds.append(batch_proto[i][idx,:])

    # embeds = torch.stack(batch_proto).cuda()
    embeds = torch.stack(batch_proto).to(device)
    embeds = embeds.unsqueeze(1)
    # print(embeds.shape)
    return embeds
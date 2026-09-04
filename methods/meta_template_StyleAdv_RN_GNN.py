import os
import torch.nn as nn
import torch
import numpy as np
import time
from abc import abstractmethod
from tensorboardX import SummaryWriter
from methods.tool_func import consistency_loss
from utils.distributed_utils import average_gradients, is_main_process, reduce_scalar, sync_model_buffers
try:
  from tqdm import tqdm
except ImportError:  # pragma: no cover
  tqdm = None

 
class MetaTemplate(nn.Module):
  def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True, device=None):
    super(MetaTemplate, self).__init__()
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    if tf_path is not None and is_main_process():
      resolved_tf_path = tf_path
      if os.path.isfile(resolved_tf_path):
        resolved_tf_path = resolved_tf_path + "_tb"
      elif os.path.isdir(resolved_tf_path):
        resolved_tf_path = os.path.join(
          resolved_tf_path,
          f"events_{time.time_ns()}_{os.getpid()}",
        )
      self.tf_writer = SummaryWriter(log_dir=resolved_tf_path)
    else:
      self.tf_writer = None
    self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move module parameters and buffers to the chosen device (CPU if no GPU)
    try:
      self.to(self.device)
    except Exception:
      # fallback: some environments may not allow module.to at init; caller can move later
      pass

  @abstractmethod
  def set_forward(self, x, is_feature=False, global_y=None):
    pass

  @abstractmethod
  def set_forward_loss(self, x, global_y=None):
    pass

  def forward(self,x):
    out  = self.feature.forward(x)
    return out

  def parse_feature(self,x,is_feature):
    x = x.to(self.device, non_blocking=True)
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def correct(self, x, global_y=None):
    scores, loss = self.set_forward_loss(x, global_y=global_y)
    y_query = np.repeat(range( self.n_way ), self.n_query )

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)


  def train_loop(
    self,
    epoch,
    train_loader_ori,
    optimizer,
    total_it,
    target_unlabeled_loader=None,
    target_ssl_weight=0.0,
    target_ssl_temperature=0.5,
    target_ssl_ramp_epochs=60,
  ):
    print_freq = max(1, len(train_loader_ori) // 10)
    avg_loss=0
    num_batches = 0
    avg_target_loss = 0.0
    avg_target_conf = 0.0
    avg_target_entropy = 0.0
    avg_target_keep_ratio = 0.0
    avg_target_selected_conf = 0.0
    avg_target_view = 0.0
    avg_target_selected_view = 0.0
    num_target_batches = 0
    avg_data_time = 0.0
    avg_step_time = 0.0
    iterator = tqdm(train_loader_ori, desc=f"Epoch {epoch}", leave=False) if tqdm is not None and is_main_process() else train_loader_ori
    last_end_time = time.perf_counter()
    target_unlabeled_iter = iter(target_unlabeled_loader) if target_unlabeled_loader is not None else None
    for i, (x_ori, global_y ) in enumerate(iterator):
      data_ready_time = time.perf_counter()
      avg_data_time += data_ready_time - last_end_time
      step_start_time = data_ready_time
      self.n_query = x_ori.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x_ori.size(0)
      optimizer.zero_grad()

      epsilon_list = getattr(self, 'attack_epsilon_list', [0.8, 0.08, 0.008])

      scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv = self.set_forward_loss_StyAdv(x_ori, global_y, epsilon_list)

      # consistency loss between initial and styleAdv
      if(scores_fsl_ori.equal(scores_fsl_adv)):
        loss_fsl_KL = 0
      else:
        loss_fsl_KL = consistency_loss(scores_fsl_ori, scores_fsl_adv, 'KL3')
      
      if(scores_cls_ori.equal(scores_cls_adv)):
        loss_cls_KL = 0
      else:
        loss_cls_KL = consistency_loss(scores_cls_ori, scores_cls_adv,'KL3')
      
  
      # final loss 
      k1, k2, k3, k4, k5, k6 = 1, 1, 1, 1, 0, 0     
      loss = k1 * loss_fsl_ori + k2 * loss_fsl_adv + k3 * loss_fsl_KL + k4 * loss_cls_ori + k5 * loss_cls_adv + k6 * loss_cls_KL

      target_loss = None
      target_stats = {}
      target_weight = 0.0
      if (
        target_unlabeled_iter is not None
        and target_ssl_weight > 0
        and hasattr(self, 'compute_target_unlabeled_loss')
      ):
        try:
          x_target_weak, x_target_strong = next(target_unlabeled_iter)
        except StopIteration:
          target_unlabeled_iter = iter(target_unlabeled_loader)
          x_target_weak, x_target_strong = next(target_unlabeled_iter)

        target_weight = float(target_ssl_weight)
        if target_ssl_ramp_epochs > 0:
          target_weight *= min(1.0, float(epoch + 1) / float(target_ssl_ramp_epochs))

        target_loss, target_stats = self.compute_target_unlabeled_loss(
          x_target_weak,
          x_target_strong,
          temperature=target_ssl_temperature,
        )
        if hasattr(self, 'adjust_target_ssl_weight'):
          target_weight, adaptive_factor = self.adjust_target_ssl_weight(target_weight, target_stats)
          target_stats['weight_factor'] = adaptive_factor
        loss = loss + target_weight * target_loss

      loss.backward()
      if getattr(self, 'distributed', False):
        average_gradients(self)
      optimizer.step()
      if getattr(self, 'distributed', False):
        sync_model_buffers(self)
      avg_loss = avg_loss+loss.item()
      num_batches += 1
      if target_loss is not None:
        avg_target_loss += target_loss.item()
        avg_target_conf += float(target_stats.get('pseudo_confidence', 0.0))
        avg_target_entropy += float(target_stats.get('mean_entropy', 0.0))
        avg_target_keep_ratio += float(target_stats.get('selected_ratio', 0.0))
        avg_target_selected_conf += float(target_stats.get('selected_confidence', 0.0))
        avg_target_view += float(target_stats.get('view_agreement', 0.0))
        avg_target_selected_view += float(target_stats.get('selected_view_agreement', 0.0))
        num_target_batches += 1
      avg_step_time += time.perf_counter() - step_start_time

      if (i + 1) % print_freq==0 and tqdm is None and is_main_process():
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader_ori), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar('loss_fsl_ori:', loss_fsl_ori.item(), total_it +1)
        self.tf_writer.add_scalar('loss_fsl_adv:', loss_fsl_adv.item(), total_it +1)
        #self.tf_writer.add_scalar('loss_fsl_KL:', loss_fsl_KL.item(), total_it +1)
        self.tf_writer.add_scalar('loss_cls_ori:', loss_cls_ori.item(), total_it +1)
        self.tf_writer.add_scalar('loss_cls_adv:', loss_cls_adv.item(), total_it +1)
        #self.tf_writer.add_scalar('loss_cls_Kl:', loss_cls_KL.item(), total_it +1)
        self.tf_writer.add_scalar('total_loss:', loss.item(), total_it +1)
        # intial
        self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)
        if hasattr(self, 'last_text_guidance_weight'):
          self.tf_writer.add_scalar('text_guidance_weight', getattr(self, 'last_text_guidance_weight', 1.0), total_it + 1)
        if hasattr(self, 'last_text_prompt_agreement'):
          self.tf_writer.add_scalar('text_prompt_agreement', getattr(self, 'last_text_prompt_agreement', 1.0), total_it + 1)
        if hasattr(self, 'last_semantic_anchor_loss'):
          self.tf_writer.add_scalar(
            'semantic_anchor_loss',
            getattr(self, 'last_semantic_anchor_loss', 0.0),
            total_it + 1,
          )
        semantic_attack_stats = getattr(self, 'last_semantic_attack_stats', {})
        for stage_name, stage_stats in semantic_attack_stats.items():
          self.tf_writer.add_scalar(
            f'semantic_attack/{stage_name}/drift',
            stage_stats.get('drift', 0.0),
            total_it + 1,
          )
          self.tf_writer.add_scalar(
            f'semantic_attack/{stage_name}/lambda_before',
            stage_stats.get('lambda_before', 0.0),
            total_it + 1,
          )
          self.tf_writer.add_scalar(
            f'semantic_attack/{stage_name}/lambda_after',
            stage_stats.get('lambda_after', 0.0),
            total_it + 1,
          )
          for drift_name in (
            'support_drift',
            'query_drift',
            'positive_drift_ratio',
            'sample_drift_mean',
            'sample_drift_std',
            'sample_drift_min',
            'sample_drift_max',
          ):
            self.tf_writer.add_scalar(
              f'semantic_attack/{stage_name}/{drift_name}',
              stage_stats.get(drift_name, 0.0),
              total_it + 1,
            )
          for statistic_name in ('mean_abs_delta', 'linf_delta', 'budget_utilization'):
            self.tf_writer.add_scalar(
              f'semantic_attack/{stage_name}/mu/{statistic_name}',
              stage_stats.get('mean', {}).get(statistic_name, 0.0),
              total_it + 1,
            )
            self.tf_writer.add_scalar(
              f'semantic_attack/{stage_name}/sigma/{statistic_name}',
              stage_stats.get('sigma', {}).get(statistic_name, 0.0),
              total_it + 1,
            )
          self.tf_writer.add_scalar(
            f'semantic_attack/{stage_name}/sigma/sigma_clamp_rate',
            stage_stats.get('sigma', {}).get('sigma_clamp_rate', 0.0),
            total_it + 1,
          )
        if target_loss is not None:
          self.tf_writer.add_scalar('target_ssl_loss', target_loss.item(), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_weight', target_weight, total_it + 1)
          self.tf_writer.add_scalar('target_pseudo_confidence', target_stats.get('pseudo_confidence', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_mean_entropy', target_stats.get('mean_entropy', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_keep_ratio', target_stats.get('selected_ratio', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_selected_confidence', target_stats.get('selected_confidence', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_view_agreement', target_stats.get('view_agreement', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_selected_view_agreement', target_stats.get('selected_view_agreement', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_selected_entropy', target_stats.get('selected_entropy', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_prob_loss', target_stats.get('prob_loss', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_feature_loss', target_stats.get('feature_loss', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_prototype_loss', target_stats.get('prototype_loss', 0.0), total_it + 1)
          self.tf_writer.add_scalar('target_ssl_weight_factor', target_stats.get('weight_factor', 1.0), total_it + 1)
         
      total_it += 1
      last_end_time = time.perf_counter()
    mean_loss = avg_loss / float(num_batches or 1)
    mean_loss = reduce_scalar(mean_loss, self.device, average=True)
    mean_target_loss = reduce_scalar(avg_target_loss / float(num_target_batches or 1), self.device, average=True)
    mean_target_conf = reduce_scalar(avg_target_conf / float(num_target_batches or 1), self.device, average=True)
    mean_target_entropy = reduce_scalar(avg_target_entropy / float(num_target_batches or 1), self.device, average=True)
    mean_target_keep_ratio = reduce_scalar(avg_target_keep_ratio / float(num_target_batches or 1), self.device, average=True)
    mean_target_selected_conf = reduce_scalar(avg_target_selected_conf / float(num_target_batches or 1), self.device, average=True)
    mean_target_view = reduce_scalar(avg_target_view / float(num_target_batches or 1), self.device, average=True)
    mean_target_selected_view = reduce_scalar(avg_target_selected_view / float(num_target_batches or 1), self.device, average=True)
    mean_data_time = reduce_scalar(avg_data_time / float(num_batches or 1), self.device, average=True)
    mean_step_time = reduce_scalar(avg_step_time / float(num_batches or 1), self.device, average=True)
    if num_batches > 0 and is_main_process():
      print(
        'Epoch {:d} | Mean Loss {:f} | target_ssl {:.6f} | pseudo_conf {:.4f} | mean_entropy {:.4f} | keep_ratio {:.4f} | selected_conf {:.4f} | view_agree {:.4f} | selected_view {:.4f} | data_time {:.4f}s | step_time {:.4f}s'.format(
          epoch,
          mean_loss,
          mean_target_loss,
          mean_target_conf,
          mean_target_entropy,
          mean_target_keep_ratio,
          mean_target_selected_conf,
          mean_target_view,
          mean_target_selected_view,
          mean_data_time,
          mean_step_time,
        )
      )
    return total_it, mean_loss

  def test_loop(self, test_loader, record = None, epoch = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    iterator = tqdm(test_loader, desc="Evaluating", leave=False) if tqdm is not None and is_main_process() else test_loader
    for i, (x,global_y) in enumerate(iterator):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct(x, global_y=global_y)
      acc_all.append(correct_this/ count_this*100  )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    epoch_str = f'Epoch {epoch} | ' if epoch is not None else ''
    print(f'{epoch_str}--- {iter_num} Loss = {loss/count:.6f} ---')
    print(f'{epoch_str}--- {iter_num} Test Acc = {acc_mean:4.2f}% +- {1.96* acc_std/np.sqrt(iter_num):4.2f}% ---')

    return acc_mean

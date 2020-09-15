import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.optim as optim
import torch.utils.data
import datetime
from utils.utilities import (create_folder, get_filename, create_logging, Mixup,
    StatisticsContainer)
from pytorch.models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout,
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch.pytorch_utils import (move_data_to_device, count_parameters, count_flops,
    do_mixup)
from utils.data_generator import (AudioSetDataset, TrainSampler, BalancedTrainSampler,
    AlternateTrainSampler, EvaluateSampler, collate_fn)
from pytorch.evaluate import Evaluator
from birds import config
from pytorch.losses import get_loss_func
from pytorch.finetune_template import Transfer_Cnn14


def eval_model(evaluator, iteration, statistics_container,
               eval_bal_loader, train_bgn_time, device, loss_func):
    # Evaluate

    train_fin_time = time.time()

    bal_statistics, output_dict = evaluator.evaluate(eval_bal_loader)
    # test_statistics = evaluator.evaluate(eval_test_loader)

    logging.info('Validate bal average_precision: {:.3f}'.format(
        np.mean(bal_statistics['average_precision'])))
    logging.info('Validate bal auc: {:.3f}'.format(
        np.mean(bal_statistics['auc'])))
    try:
        logging.info('Validate bal micro_f1: {:.3f}'.format(
            bal_statistics['micro_f1']))
    except KeyError:
        pass
    for key in output_dict.keys():
        if key in ['target', 'clipwise_output']:
            output_dict[key] = move_data_to_device(torch.from_numpy(output_dict[key]), device)
    val_loss = loss_func(output_dict, output_dict)

    logger = logging.getLogger()
    logger.handlers[0].flush()

    # logging.info('Validate test micro_f1: {:.3f}'.format(
    #     np.mean(test_statistics['micro_f1'])))

    statistics_container.append(iteration, bal_statistics, data_type='bal')
    # statistics_container.append(iteration, test_statistics, data_type='test')
    statistics_container.dump()

    train_time = train_fin_time - train_bgn_time
    validate_time = time.time() - train_fin_time

    logging.info(
        'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
        ''.format(iteration, train_time, validate_time))

    logging.info('------------------------------------')

    train_bgn_time = time.time()
    return train_bgn_time, val_loss


def train_epoch(model, train_loader, iteration, train_sampler, checkpoints_dir,
                augmentation, device, loss_func, train_log_interval, eval_interval,
                save_interval, optimizer):
    model.train()
    time1 = time.time()
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Save model
        if iteration and iteration % save_interval == 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.module.state_dict(),
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'],
                                      batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                    batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target'].float()}
            """{'target': (batch_size, classes_num)}"""
        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        print(f'loss: {loss.data}')
        logging.info(f'loss: {loss.data}')
        # Backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if iteration % train_log_interval == 0:
            print('--- Iteration: {}, train time: {:.3f} s / {} iterations ---'.format(
                iteration, time.time() - time1, train_log_interval))
        iteration += 1
        if iteration % eval_interval == 0:
            return iteration, loss, optimizer, model


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str+
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    load_pretrained = args.load_pretrained
    train_log_interval = args.train_log_interval
    eval_interval = args.eval_interval
    save_interval = args.save_interval

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    # Paths
    black_list_csv = None

    train_indexes_hdf5_path = os.path.join(workspace, 'train_indices')

    if args.mini_val != 'none':
        eval_bal_indexes_hdf5_path = args.mini_val
    else:
        eval_bal_indexes_hdf5_path = os.path.join(workspace, 'val_indices')

    # eval_test_indexes_hdf5_path = os.path.join(workspace, 'val_indices')

    now = datetime.datetime.now().strftime("%m/%d:%-H:%-M")
    checkpoints_dir = os.path.join(workspace, now, 'checkpoints', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                                   'data_type={}'.format(data_type), model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, now, 'statistics', filename,
                                   'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                                   'data_type={}'.format(data_type), model_type,
                                   'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                                   'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
                                   'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, now, 'logs', filename,
                            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                                sample_rate, window_size, hop_size, mel_bins, fmin, fmax),
                            'data_type={}'.format(data_type), model_type,
                            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced),
                            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'

    if load_pretrained:
        try:
            model = Transfer_Cnn14(sample_rate=sample_rate, window_size=window_size,
                                   hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                                   fine_tune_classes_num=classes_num, model_type=model_type)
            checkpoint = torch.load(load_pretrained, map_location=device)
            model.load_from_pretrain(checkpoint['model'])
            logging.info(f'loaded pretrained model from {load_pretrained}')
        except Exception as e:
            print(e)
            model = Transfer_Cnn14(sample_rate=sample_rate, window_size=window_size,
                                   hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                                   fine_tune_classes_num=classes_num, model_type=model_type, freeze_base=False)
            logging.info('started Transfer_Cnn14 model from fresh')
        if 'cuda' in str(device):
            model.to(device)

    else:
        # Model
        Model = eval(model_type)
        model = Model(sample_rate=sample_rate, window_size=window_size,
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                      classes_num=classes_num)

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(clip_samples=clip_samples, classes_num=classes_num)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv)
    
    # Evaluate sampler
    eval_bal_sampler = Sampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=16)

    # eval_test_sampler = EvaluateSampler(
    #     indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # eval_test_loader = torch.utils.data.DataLoader(dataset=dataset,
    #     batch_sampler=eval_test_sampler, collate_fn=collate_fn,
    #     num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = Evaluator(model=model)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, eps=args.eps)
    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    val_losses = []
    layers = [child for name, child in model.module.base.named_children() if name not in ['spectrogram_extractor', 'logmel_extractor']]
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in layers[-1].parameters():
    #     param.requires_grad = True
    # layers = layers[:-1]
    if args.unfreeze_all:
        for param in model.module.base.parameters():
            param.require_grad = True
    names = [name for name, child in model.module.base.named_children() if name not in ['spectrogram_extractor', 'logmel_extractor']]
    unfreeze = len(layers) - 1
    strikes = 0
    lr = optimizer.param_groups[0]['lr']
    for epoch in range(resume_iteration + 1, early_stop + 1):
        train_bgn_time, val_loss = eval_model(evaluator, iteration, statistics_container, eval_bal_loader,
                                              train_bgn_time, device, loss_func)
        if len(val_losses):
            if val_loss >= val_losses[-1] * 0.95:
                strikes += 1
                if unfreeze > -1 and strikes > 3 and not args.unfreeze_all:
                    for name, child in model.module.base.named_children():
                        if name == names[unfreeze]:
                            for param in child.parameters():
                                param.requires_grad = True
                            logging.info(f'unfroze {names[unfreeze]}')
                            unfreeze -= 1
                            strikes = 0
        val_losses.append(val_loss)
        if len(val_losses) > 10:
            val_losses = val_losses[-10:]
        # if len(val_losses > )
        scheduler.step(val_loss)
        updated_lr = optimizer.param_groups[0]['lr']
        if updated_lr != lr:
            logging.info(f'learning rate updated from {lr} to {updated_lr}')
            lr = updated_lr

        iteration, loss, optimizer, model = train_epoch(model, train_loader, iteration, train_sampler, checkpoints_dir,
                                                        augmentation, device, loss_func, train_log_interval, eval_interval,
                                                        save_interval, optimizer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, default='/home/ubuntu/audioset_tagging_cnn/birds/full')
    # parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--data_type', type=str, default='mini', choices=['mini', 'full'])
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000)
    parser_train.add_argument('--train_log_interval', type=int, default=10)
    parser_train.add_argument('--eval_interval', type=int, default=100)
    parser_train.add_argument('--save_interval', type=int, default=500)
    parser_train.add_argument('--model_type', type=str, default='Wavegram_Logmel_Cnn14')
    parser_train.add_argument('--load_pretrained', type=str)
    parser_train.add_argument('--loss_type', type=str, default='cross_entropy', choices=['binary_cross_entropy', 'cross_entropy'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='none', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=1024)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=100)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_val', type=str, default='/home/ubuntu/audioset_tagging_cnn/birds/mini_1/val_indices',
                              choices=['/home/ubuntu/audioset_tagging_cnn/birds/mini_1/val_indices', 'none'])
    parser_train.add_argument('--factor', type=float, default=.5)
    parser_train.add_argument('--patience', type=int, default=5)
    parser_train.add_argument('--eps', type=float, default=2e-5)
    parser_train.add_argument('--unfreeze_all', action='store_true', default=False)
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    train(args)

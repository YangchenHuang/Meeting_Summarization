import argparse
import glob
import os
import random
import signal
import re

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset, load_test
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']
rouge_list = ['rouge_1_f_score', 'rouge_1_precision', 'rouge_1_recall', 'rouge_2_f_score', 'rouge_2_precision',
              'rouge_2_recall', 'rouge_l_f_score', 'rouge_l_precision', 'rouge_l_recall']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger(args.log_file)

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in Distributed initialization")

        train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def wait_and_validate(args, device_id):
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    total_loss_lst = []
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        total_loss = validate(args, device_id, cp, step)
        total_loss_lst.append((total_loss, cp))
        max_step = total_loss_lst.index(min(total_loss_lst))
        if i - max_step > 10:
            break
    total_loss_lst = sorted(total_loss_lst, key=lambda x: x[0])[:3]
    logger.info('LOSS %s' % str(total_loss_lst))


def validate(args, device_id, pt, step):
    device = "cpu" if args.world_size == 0 else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    model = Summarizer(args, device)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False), args.batch_size, device,
                                        shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.total_loss


def test(args, device_id, pt, step, is_valid=False):
    device = "cpu" if args.world_size == 0 else "cuda"
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    model = Summarizer(args, device)
    model.load_cp(checkpoint)
    model.eval()
    trainer = build_trainer(args, device_id, model, None)

    if args.test_txt:
        pts = sorted(glob.glob(args.bert_data_path + '*.pt'))
        for pt in pts:
            id = re.sub(args.bert_data_path[:-1] + '\\\\test.', '', pt)
            id = re.sub('.bert.pt', '', id)
            test_iter = data_loader.Dataloader(args, load_test(pt, 'test'),
                                               args.test_batch_size, device,
                                               shuffle=False, is_test=True)
            stats = trainer.test(test_iter, step, 'test', id)
    else:
        corpus_list = ['train', 'valid', 'test']
        if is_valid:
            corpus_list = ['valid']
        for corpus_type in corpus_list:
            rouge_stat = {}
            for type in rouge_list:
                rouge_stat[type] = []
            pts = sorted(glob.glob(args.bert_data_path + corpus_type + '*.pt'))
            for pt in pts:
                id = re.sub(args.bert_data_path[:-1] + '\\\\' + corpus_type + '.', '', pt)
                id = re.sub('.bert.pt', '', id)
                test_iter = data_loader.Dataloader(args, load_test(pt, corpus_type),
                                                   args.test_batch_size, device,
                                                   shuffle=False, is_test=True)
                stats, rouges = trainer.test(test_iter, step, corpus_type, id)
                for type in rouge_list:
                    rouge_stat[type].append(rouges[type])
            for type in rouge_list:
                rouge_stat[type] = sum(rouge_stat[type]) / len(rouge_stat[type])
                logger.info('Rouges at step %d in %s %s\n%s' % (step, corpus_type, type, rouge_stat[type]))
                # cuda0 = torch.device('cuda:0')
                # cand = torch.tensor(cand, device=cuda0)
                # mask_can = torch.tensor(mask_can, device=cuda0)
                # torch.save(sents_vec, 'document.pt')
                # torch.save(mask_cls, 'mask_cls.pt')
                # torch.save(cand, 'cand.pt')
                # torch.save(mask_can, 'mask_can.pt')
                # # model_doc = TransformerSenEncoder((768, 128, 8, 0.1, 2))
                # # print(model_doc())
                # print(sents_vec.shape, mask_cls.shape, cand.shape, mask_can.shape)


def train(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.world_size == 0 else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = Summarizer(args, device)
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, model, checkpoint)
    else:
        optim = model_builder.build_optim(args, model, None)

    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)
    for i, batch in enumerate(train_iter_fct()):
        print(i, batch)
    trainer.train(train_iter_fct, args.train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_txt", type=str2bool, nargs='?', const=False,
                        default=False)  # remember to reset all paths
    parser.add_argument("-bert_data_path", default='../ext_data/bert_data/')
    parser.add_argument("-model_path", default='../models/extract/')
    parser.add_argument("-result_path", default='../ext_data/summary/')
    parser.add_argument("-story_path", default='../ext_data/result_story/')
    parser.add_argument("-temp_dir", default='../temp/')

    parser.add_argument("-batch_size", default=1000, type=int)
    parser.add_argument("-test_batch_size", default=20000, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("-ff_size", default=512, type=int, help='feed forward network hidden size')
    parser.add_argument("-heads", default=16, type=int)
    parser.add_argument("-inter_layers", default=16, type=int)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=2e-7, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-decay_step", default=1000, type=int)

    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=5, type=int)
    parser.add_argument("-world_size", default=1, type=int, help='gpu world size, 0 if cpu')
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=5000, type=int)

    parser.add_argument('-visible_gpus', default='', type=str)
    parser.add_argument('-gpu_ranks', default='', type=str)
    parser.add_argument('-log_file', default='../logs/extractive.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')

    args = parser.parse_args()
    if args.visible_gpus == '' and args.world_size != 0:
        args.visble_gpus = [str(i) for i in range(args.world_size)]
        args.visble_gpus = ','.join(args.visble_gpus)
    if args.gpu_ranks == '' and args.world_size != 0:
        args.gpu_ranks = [str(i) for i in range(args.world_size)]
        args.gpu_ranks = ','.join(args.gpu_ranks)
    if args.gpu_ranks != '':
        args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.world_size == 0 else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.world_size > 1:
        multi_main(args)
    elif args.mode == 'train':
        train(args, device_id)
    elif args.mode == 'validate':
        wait_and_validate(args, device_id)
    elif args.mode == 'test':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test(args, device_id, cp, step)

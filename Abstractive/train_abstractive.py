from __future__ import division

import argparse
import glob
import os
import random
import signal

import torch
from transformers import BertTokenizer, LongformerTokenizer

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.loss import abs_loss
from models.model_builder import AbsSummarizer_longformer, AbsSummarizer_Bert
from models.predictor import build_predictor
from models.trainer import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_abs_multi(args):
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
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_abs_single(args, device_id)
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
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_abs(args, device_id):
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    xent_lst = []
    rouge_1_lst = []
    rouge_2_lst = []
    rouge_l_lst = []
    if args.validate_rouge:
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent, rouges = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            rouge_1_lst.append((rouges['rouge_1_f_score'], cp))
            rouge_2_lst.append((rouges['rouge_2_f_score'], cp))
            rouge_l_lst.append((rouges['rouge_l_f_score'], cp))
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        rouge_1_lst = sorted(rouge_1_lst, key=lambda x: x[0])[:5]
        rouge_2_lst = sorted(rouge_2_lst, key=lambda x: x[0])[:5]
        rouge_l_lst = sorted(rouge_l_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        logger.info('Rouge_1 %s' % str(rouge_1_lst))
        logger.info('Rouge_2 %s' % str(rouge_2_lst))
        logger.info('Rouge_l %s' % str(rouge_l_lst))
    else:
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent, rouges = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))


def validate(args, device_id, pt, step):
    device = "cpu" if args.world_size == 0 else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    if args.encoder == 'longformer':
        model = AbsSummarizer_longformer(args, device)
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', do_lower_case=True,
                                                        cache_dir=args.temp_dir)
        special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'additional_special_tokens': ['<q>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        symbols = {'BOS': tokenizer.convert_tokens_to_ids('<bos>'), 'EOS': tokenizer.convert_tokens_to_ids('<eos>'),
                   'PAD': tokenizer.convert_tokens_to_ids('<pad>'), 'EOQ': tokenizer.convert_tokens_to_ids('<q>')}
        model.longformer.model.resize_token_embeddings(len(tokenizer))
        model.embedding_resize()
        logger.info(model)
        model.load_state_dict(checkpoint['model'], strict=True)

    else:
        model = AbsSummarizer_Bert(args, device, checkpoint)
        logger.info(model)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    valid_loss = abs_loss(model.generator, symbols, model.vocab_size, train=False, device=device)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    test_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    if args.validate_rouge:
        predictor = build_predictor(args, tokenizer, symbols, model, logger)
        rouges = predictor.translate(test_iter, step, corpus_type='valid')
        return stats.xent(), rouges
    else:
        return stats.xent(), None


def test_abs(args, device_id, pt, step):
    device = "cpu" if args.world_size == 0 else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)

    if args.encoder == 'longformer':
        model = AbsSummarizer_longformer(args, device)
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', do_lower_case=True,
                                                        cache_dir=args.temp_dir)
        special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'additional_special_tokens': ['<q>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        symbols = {'BOS': tokenizer.convert_tokens_to_ids('<bos>'), 'EOS': tokenizer.convert_tokens_to_ids('<eos>'),
                   'PAD': tokenizer.convert_tokens_to_ids('<pad>'), 'EOQ': tokenizer.convert_tokens_to_ids('<q>')}
        model.longformer.model.resize_token_embeddings(len(tokenizer))
        model.embedding_resize()
        logger.info(model)
        model.load_state_dict(checkpoint['model'], strict=True)

    else:
        model = AbsSummarizer_Bert(args, device, checkpoint)
        logger.info(model)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)


def train_abs(args, device_id):
    if (args.world_size > 1):
        train_abs_multi(args)
    else:
        train_abs_single(args, device_id)


def train_abs_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.world_size == 0 else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    if args.encoder == 'longformer':
        model = AbsSummarizer_longformer(args, device)
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', do_lower_case=True,
                                                        cache_dir=args.temp_dir)
        special_tokens_dict = {'bos_token': '<bos>', 'eos_token': '<eos>', 'additional_special_tokens': ['<q>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        symbols = {'BOS': tokenizer.convert_tokens_to_ids('<bos>'), 'EOS': tokenizer.convert_tokens_to_ids('<eos>'),
                   'PAD': tokenizer.convert_tokens_to_ids('<pad>'), 'EOQ': tokenizer.convert_tokens_to_ids('<q>')}
        model.longformer.model.resize_token_embeddings(len(tokenizer))
        model.embedding_resize()
        logger.info(model)
        if args.train_from != '':
            model.load_state_dict(checkpoint['model'], strict=True)

    else:
        model = AbsSummarizer_Bert(args, device, checkpoint)
        logger.info(model)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    if (args.sep_optim):
        optim_enc = model_builder.build_optim_enc(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_enc, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    train_loss = abs_loss(model.generator, symbols, model.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'longformer'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-test_txt", type=str2bool, nargs='?', const=False,
                        default=False)  # remember to reset all paths

    parser.add_argument("-data_path", default='../abs_data/bert_data/')
    parser.add_argument("-model_path", default='../models/abstract/')
    parser.add_argument("-result_path", default='../abs_data/result/')
    parser.add_argument("-summary_path", default='../result_summary/')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=200, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-lr", default=2e-3, type=float)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=500, type=int)
    parser.add_argument("-warmup_steps_enc", default=500, type=int)
    parser.add_argument("-warmup_steps_dec", default=500, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.1, type=float)
    parser.add_argument("-beam_size", default=10, type=int)
    parser.add_argument("-min_length", default=280, type=int)
    parser.add_argument("-max_length", default=400, type=int)
    parser.add_argument("-max_tgt_len", default=512, type=int)

    parser.add_argument("-save_checkpoint_steps", default=500, type=int)
    parser.add_argument("-accum_count", default=5, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=2000, type=int)
    parser.add_argument("-validate_rouge", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-world_size", default=1, type=int, help='gpu world size, 0 if cpu')
    parser.add_argument('-visible_gpus', default='', type=str)
    parser.add_argument('-gpu_ranks', default='', type=str)
    parser.add_argument('-log_file', default='../logs/abstractive.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_from", default='')

    parser.add_argument("-train_from", default='')
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    if args.visible_gpus == '' and args.world_size != 0:
        args.visble_gpus = [str(i) for i in range(args.world_size)]
        args.visble_gpus = ','.join(args.visble_gpus)
    if args.gpu_ranks == '' and args.world_size != 0:
        args.gpu_ranks = [str(i) for i in range(args.world_size)]
        args.gpu_ranks = ','.join(args.gpu_ranks)
    if args.gpu_ranks != '':
        args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.world_size == 0 else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.mode == 'train':
        train_abs(args, device_id)
    elif args.mode == 'validate':
        validate_abs(args, device_id)
    if args.mode == 'test':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_abs(args, device_id, cp, step)

import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from others.utils import calculate_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Trainer creation

    """
    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if model:
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        self.weight_loss = torch.nn.MSELoss()
        assert grad_accum_count > 0

        if model:
            self.model.train()

    def train(self, train_iter_fct, train_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            train_steps(int):

        Return:
            train_stats
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)
                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0:
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls
                weight = batch.weight

                sents_vec, sent_scores, mask, cluster_weight = self.model(src, segs, clss, mask, mask_cls)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                weight_loss = self.weight_loss(weight, cluster_weight)
                total_loss = loss + weight_loss * 10
                # print(weight, cluster_weight)
                # print(weight_loss)
                batch_stats = Statistics(float(total_loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, corpus_type, id):
        """ Test model and generate extractive summary.
            test_iter: validate data iterator
            step: checkpoint step
            corpus_type: train/test/validate
            id: the document to process
        Returns:
            stat: loss statistics
            rouges: rouge statistics of the document
        """

        self.model.eval()
        stats = Statistics()
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        if not os.path.exists(self.args.story_path):
            os.makedirs(self.args.story_path)
        can_path = self.args.result_path + corpus_type + '.' + id + '_step%d.candidate' % step
        gold_path = self.args.result_path + corpus_type + '.' + id + '_step%d.gold' % step
        story_path = self.args.story_path + corpus_type + '.' + id + '.story'
        with open(story_path, 'w') as save_story:
            with open(can_path, 'w') as save_pred:
                with open(gold_path, 'w') as save_gold:
                    with torch.no_grad():
                        for batch in test_iter:
                            src = batch.src
                            labels = batch.labels
                            segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask
                            mask_cls = batch.mask_cls
                            weight = batch.weight
                            index = batch.index

                            pred = []

                            sents_vec, sent_scores, mask, cluster_weight = self.model(src, segs, clss, mask, mask_cls)
                            loss = self.loss(sent_scores, labels.float())
                            weight_loss = self.weight_loss(cluster_weight, weight)
                            loss = (loss * mask.float()).sum()
                            total_loss = loss + weight_loss * 10
                            batch_stats = Statistics(float(total_loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            cluster_weight = cluster_weight.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                            cluster_weight = np.argsort(cluster_weight)
                            # print(selected_ids)
                            # selected_ids = np.sort(selected_ids,1)
                            cluster_num = len(cluster_weight)
                            for i, idx in enumerate(selected_ids):
                                rank = np.where(cluster_weight == i)[0][0]

                                if rank <= max(cluster_num // 6, 6):
                                    for j in range(5):
                                        sen_ind = selected_ids[i][j]
                                        _pred = batch.src_str[i][sen_ind].strip()
                                        pred.append((index[i][sen_ind], _pred))
                                elif rank <= max(cluster_num // 3, 10):
                                    for j in range(3):
                                        sen_ind = selected_ids[i][j]
                                        _pred = batch.src_str[i][sen_ind].strip()
                                        pred.append((index[i][sen_ind], _pred))
                                elif rank <= max(cluster_num * 2 // 3, 15):
                                    for j in range(2):
                                        sen_ind = selected_ids[i][j]
                                        _pred = batch.src_str[i][sen_ind].strip()
                                        pred.append((index[i][sen_ind], _pred))
                                else:
                                    sen_ind = selected_ids[i][0]
                                    _pred = batch.src_str[i][sen_ind].strip()
                                    pred.append((index[i][sen_ind], _pred))

                            gold_summary = (batch.tgt_str[0].strip())
                            pred.sort(key=lambda x: x[0])
                            for i in range(len(pred)):
                                save_story.write(pred[i][1].strip() + '\n')
                                if i == 0:
                                    save_pred.write(pred[i][1].strip())
                                else:
                                    save_pred.write('<q> ' + pred[i][1].strip())
                    save_gold.write(gold_summary)
                for sent in gold_summary.split('<q>'):
                    save_story.write('@highlight {}\n'.format(sent))
        if self.args.test_txt:
            return stats
        else:
            rouges = calculate_rouge(can_path, gold_path)
            logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            self._report_step(0, step, valid_stats=stats)
            return stats, rouges



    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls
            weight = batch.weight

            sents_vec, sent_scores, mask, cluster_weight = self.model(src, segs, clss, mask, mask_cls)
            # print(sent_scores, labels)
            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            weight_loss = self.weight_loss(cluster_weight, weight)
            total_loss = loss + weight_loss * 10
            (total_loss / total_loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(total_loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)

        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

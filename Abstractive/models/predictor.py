import codecs
import re
import os

import torch

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Summarize meetings using this translator
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = self.vocab.convert_tokens_to_ids('<bos>')
        self.end_token = self.vocab.convert_tokens_to_ids('<eos>')

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src, id = translation_batch["predictions"], \
                                                          translation_batch["scores"], translation_batch["gold_score"], \
                                                          batch.tgt_str, batch.src, batch.id

        translations = []
        for b in range(batch_size):
            pred_ids = [int(n) for n in preds[b][0]]
            print(pred_ids)
            pred_sents = self.vocab.decode(pred_ids)
            print(pred_sents)
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = self.vocab.decode([int(t) for t in src[b]])
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src, id[b])
            translations.append(translation)

        return translations

    def translate(self, data_iter, step, corpus_type='test'):

        self.model.eval()
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        if not os.path.exists(self.args.summary_path):
            os.makedirs(self.args.summary_path)

        if not self.args.test_txt:
            gold_path = self.args.result_path + corpus_type + '.%d.gold' % step
            can_path = self.args.result_path + corpus_type + '.%d.candidate' % step
            raw_src_path = self.args.result_path + '.%d.raw_src' % step
            self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
            self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

            self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
            self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
            self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                with torch.no_grad():
                    batch_data = self.translate_batch(batch, self.max_length, self.min_length)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src, id = trans
                    if self.args.encoder == 'longformer':
                        pred_str = pred.replace('[bos]', '').replace('[pad]', '').replace('[eos]', '').replace \
                            (r' +', ' ').strip()
                    elif self.args.encoder == 'bert':
                        for unused in ['\[unused1\]', '\[unused4\]', '\[unused5\]', '\[unused6\]', '\[unused7\]',
                                       '\[unused8\]',
                                       '\[unused9\]', '\[unused10\]', '\[unused11\]', '\[unused12\]', '\[unused13\]',
                                       '\[unused14\]']:
                            pred = re.sub(unused + '.*', '', pred)
                        pred_str = pred.replace('[unucsed0]', '').replace('[unused3]', '').replace('[PAD]', '') \
                            .replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                    pred_str = re.sub('\\b(\\w+)\\s+\\1\\b', '\\1', pred_str)
                    pred_str = re.sub('(\\b.+?\\b)\\1\\b', '\\1', pred_str)
                    gold_str = gold.strip()

                    if corpus_type == 'test':
                        summary_path = self.args.summary_path + '%d.%s.summary' % (step, id)
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            f.write(pred_str.replace('<q>', ' '))
                    if not self.args.test_txt:
                        self.can_out_file.write(pred_str + '\n')
                        self.gold_out_file.write(gold_str + '\n')
                        self.src_out_file.write(src.strip() + '\n')
                        ct += 1

                if not self.args.test_txt:
                    self.can_out_file.flush()
                    self.gold_out_file.flush()
                    self.src_out_file.flush()

        if not self.args.test_txt:
            self.can_out_file.close()
            self.gold_out_file.close()
            self.src_out_file.close()

        rouges = {}
        if not self.args.test_txt:
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
        return rouges

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, max_length, min_length=0):

        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        mask_src = batch.mask_src

        if self.args.encoder == 'longformer':
            src_features = self.model.longformer(src, mask_src)
        elif self.args.encoder == 'bert':
            src_features = self.model.bert(src, mask_src)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)
            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            # print(curr_scores.shape)

            # cur_len = alive_seq.size(1)
            # if cur_len > 3:
            #     for i in range(alive_seq.size(0)):
            #         tokens = [int(w) for w in alive_seq[i]]
            #
            #         if len(tokens) <= 3:
            #             continue
            #         token = tokens[-1]
            #         if token in tokens[:-2]:
            #             curr_scores[i] = -10e20

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = self.vocab.decode(words)
                        words = words.split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            # if(self.args.block_trigram):
            #     cur_len = alive_seq.size(1)
            #     if(cur_len>3):
            #         for i in range(alive_seq.size(0)):
            #             tokens = [int(w) for w in alive_seq[i]]
            #             # words = self.vocab.decode(words)
            #             # words = words.split()
            #             if(len(tokens)<=3):
            #                 continue
            #
            #             trigrams = [(tokens[i-1],tokens[i],tokens[i+1]) for i in range(1,len(tokens)-1)]
            #             trigram = tuple(trigrams[-1])
            #             if trigram in trigrams[:-1]:
            #                 curr_scores[i] = -10e20
            #                 continue
            #
            #             words = self.vocab.decode(tokens)
            #             words = words.split()
            #
            #             if len(words) <= 3:
            #                 continue
            #
            #             trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
            #             trigram = tuple(trigrams[-1])
            #             if trigram in trigrams[:-1]:
            #                 curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            # print(topk_scores, topk_ids)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty
            # print(topk_log_probs)

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)
            # print(topk_ids)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            # print(alive_seq)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]
                        # print(score, pred)
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

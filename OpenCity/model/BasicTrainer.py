import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm
from lib.data_process import get_key_from_value


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict,
                 args, scheduler):
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.loss = loss
        self.optimizer = optimizer
        self.num_nodes_dict = args.num_nodes_dict
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scaler_dict = scaler_dict
        self.scheduler = scheduler
        self.batch_seen = 0
        self.best_path = os.path.join(self.args.log_dir, self.args.save_pretrain_path)
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

    def multi_train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        # train_loss_list = []
        val_loss_list = []

        for epoch in tqdm(range(self.args.epochs)):
            # Train
            # start_time = time.time()
            train_epoch_loss = self.multi_train_eps()
            # training_time = time.time() - start_time

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
                # Val
                val_epoch_loss = self.multi_val_epoch(epoch)
                # Best state and early stop epoch
                val_loss_list.append(val_epoch_loss)
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False

                # early stop
                if self.args.early_stop:
                    if not_improved_count == self.args.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.args.early_stop_patience))
                        break

                # save the best state
                if best_state == True:
                    self.logger.info('*********************************Current best model saved!')
                    # self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
                    best_model = copy.deepcopy(self.model.state_dict())
        # test
        if self.args.mode != 'pretrain' and self.args.val_ratio > 0:
            self.model.load_state_dict(best_model)
            self.test(self.model, self.args, self.scaler_dict, self.test_dataloader, self.logger)
        self.logger.info("Pre-train finish.")


    def multi_train_eps(self):
        self.model.train()
        total_loss = 0
        step = 0
        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
            select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
            out = self.model(inputs, targets, select_dataset, batch_seen=None)
            self.optimizer.zero_grad()
            loss_pred = self.loss(out, targets[..., :self.args.output_dim], self.scaler_dict[select_dataset])
            loss = loss_pred
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # learning rate decay
            if self.args.lr_decay:
                self.scheduler.step()
            total_loss += loss.item()
            step += 1
            if step % self.args.log_step == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    "step:  " + str(step) + "  train loss is:  " + str(total_loss / step) + "  current_lr is:  " + str(
                        current_lr))
            if step % self.args.save_step == 0 and self.args.debug:
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)
                self.logger.info("Saving current best model to " + self.best_path)
        train_loss = total_loss / len(self.train_dataloader)
        return train_loss

    def multi_val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.args.device), targets.squeeze(0).to(self.args.device)
                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])
                out = self.model(inputs, targets, select_dataset, batch_seen=None)
                loss_pred = self.loss(out, targets[..., 0:self.args.output_dim], self.scaler_dict[select_dataset])
                if not torch.isnan(loss_pred):
                    total_val_loss += loss_pred.item()
                val_loss = total_val_loss / len(self.val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    @staticmethod
    def test(model, args, scaler_dict, test_dataloader, logger, path=None):
        if path != None:
            if str(args.device) != 'cpu' and torch.cuda.device_count() > 1:
                model.load_state_dict(torch.load(path, map_location=args.device))
            else:
                model_weights = {k.replace('module.', ''): v for k, v in torch.load(path, map_location=args.device).items()}
                model.load_state_dict(model_weights)
            model.to(args.device)
        model.eval()

        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_batch = 0
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(args.device), targets.squeeze(0).to(args.device)
                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                output = model(inputs, targets, select_dataset, batch_seen=None)
                if args.real_value == False:
                    output = scaler_dict[select_dataset].inverse_transform(output)
                    y_lbl = scaler_dict[select_dataset].inverse_transform(targets[..., :args.output_dim])
                else:
                    y_lbl = targets[..., :args.output_dim]
                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(output, y_lbl, args.mae_thresh, args.mape_thresh)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_batch += len(y_lbl)
                logger.info(
                    f'[Test] batch {total_batch}, '
                    f'MAE={batch_mae:.4f}, RMSE={batch_rmse:.4f}, MAPE={batch_mape:.4f}, '
                    f'mae_count={mae_count}, mape_count={mape_count}'
                )
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count

        logger.info(f'[Test] === Summary ===')
        logger.info(f'[Test] total_samples={total_batch}, total_mae_count={total_count}, total_mape_count={total_mape_count}')
        logger.info("[Test] MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
            mae, rmse, mape * 100, corr))

    def _precompute_ict_cache(self, dataloader, cache_dir, desc='Precompute'):
        """Pre-compute base model outputs and save per-batch .pt files to disk.

        Since the base model is frozen, predictions and encoder features are
        constant across epochs.  Computing them once and caching to disk avoids
        redundant forward passes during aggregator training.

        Each .pt file stores a dict:
            query_pred  [B, T, N, 1]
            query_enc   [B, N, D]
            corrections [B, K, T, N, 1]
            demo_encs   [B, K, N, D]
            targets     [B, T, N, 1]
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.model.eval()
        args = self.args
        predictor = self.model.predictor

        batch_idx = 0
        max_batches = getattr(args, 'debug_batches', 0)
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=desc):
                # Resume support: skip already-cached batches
                cache_file = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
                if os.path.exists(cache_file):
                    batch_idx += 1
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    continue

                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(args.device)
                targets = targets.squeeze(0).to(args.device)
                demos_x = demos_x.squeeze(0).to(args.device)
                demos_y = demos_y.squeeze(0).to(args.device)

                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])

                # Query
                query_pred, query_enc = predictor._forward_with_features(
                    inputs, targets, select_dataset)

                # Demos
                K = demos_x.shape[1]  # demos_x: [B, S, K, T, N, F] → use S=0
                dx = demos_x[:, 0]    # [B, K, T, N, F]
                dy = demos_y[:, 0]    # [B, K, T, N, F]
                corrections_list = []
                demo_encs_list = []
                for k in range(K):
                    dk_pred, dk_enc = predictor._forward_with_features(
                        dx[:, k], dy[:, k], select_dataset)
                    corrections_list.append(dy[:, k, :, :, :args.output_dim] - dk_pred)
                    demo_encs_list.append(dk_enc)

                cache = {
                    'query_pred':  query_pred.cpu().half(),
                    'query_enc':   query_enc.cpu().half(),
                    'corrections': torch.stack(corrections_list, dim=1).cpu().half(),
                    'demo_encs':   torch.stack(demo_encs_list, dim=1).cpu().half(),
                    'targets':     targets[..., :args.output_dim].cpu().half(),
                }
                torch.save(cache, os.path.join(cache_dir, f'batch_{batch_idx}.pt'))
                batch_idx += 1
                if max_batches > 0 and batch_idx >= max_batches:
                    break

        # Count total cached batches (including previously cached)
        total_cached = len([f for f in os.listdir(cache_dir) if f.startswith('batch_') and f.endswith('.pt')])
        self.logger.info(f'[Cache] {cache_dir}: {total_cached} batches cached on disk')
        return total_cached

    def _precompute_inmemory(self, dataloader, desc='Precompute'):
        """Pre-compute base model outputs and store in CPU memory (half precision).

        Avoids disk I/O entirely.  Returns a list of dicts, one per batch:
            query_pred  [B, T, N, 1]   (cpu, float16)
            query_enc   [B, N, D]      (cpu, float16)
            corrections [B, K, T, N, 1](cpu, float16)
            demo_encs   [B, K, N, D]   (cpu, float16)
            targets     [B, T, N, 1]   (cpu, float16)
        """
        self.model.eval()
        args = self.args
        predictor = self.model.predictor
        max_batches = getattr(args, 'debug_batches', 0)

        cache_list = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=desc)):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(args.device)
                targets = targets.squeeze(0).to(args.device)
                demos_x = demos_x.squeeze(0).to(args.device)
                demos_y = demos_y.squeeze(0).to(args.device)

                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])

                # Query
                query_pred, query_enc = predictor._forward_with_features(
                    inputs, targets, select_dataset)

                # Demos
                K = demos_x.shape[1]
                dx = demos_x[:, 0]
                dy = demos_y[:, 0]
                corrections_list = []
                demo_encs_list = []
                for k in range(K):
                    dk_pred, dk_enc = predictor._forward_with_features(
                        dx[:, k], dy[:, k], select_dataset)
                    corrections_list.append(dy[:, k, :, :, :args.output_dim] - dk_pred)
                    demo_encs_list.append(dk_enc)

                cache_list.append({
                    'query_pred':  query_pred.cpu().half(),
                    'query_enc':   query_enc.cpu().half(),
                    'corrections': torch.stack(corrections_list, dim=1).cpu().half(),
                    'demo_encs':   torch.stack(demo_encs_list, dim=1).cpu().half(),
                    'targets':     targets[..., :args.output_dim].cpu().half(),
                })

        self.logger.info(f'[InMemCache] {desc}: {len(cache_list)} batches cached in RAM')
        return cache_list

    def train_aggregator(self):
        """Train DemoAggregator with in-memory caching.

        Phase 1: Run frozen base model once over train & val sets, cache all
                 intermediate outputs (query_pred, query_enc, corrections,
                 demo_encs, targets) in CPU RAM as float16.
        Phase 2: Train aggregator for multiple epochs reading from RAM only.
                 ~10-20x faster than re-running base model every batch.
        """
        import random as _random
        args = self.args

        # 1. Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # 2. Initialize aggregator
        predictor = self.model.predictor
        aggregator = predictor.init_demo_aggregator(args.aggregator_type)
        aggregator = aggregator.to(args.device)
        for param in aggregator.parameters():
            param.requires_grad = True

        # 3. Optimizer for aggregator only
        agg_optimizer = torch.optim.Adam(aggregator.parameters(), lr=args.aggregator_lr)

        self.logger.info(f'[Aggregator] type={args.aggregator_type}, '
                         f'params={sum(p.numel() for p in aggregator.parameters())}, '
                         f'lr={args.aggregator_lr}, epochs={args.aggregator_epochs}')

        # 4. Resume from checkpoint if available
        ckpt_path = os.path.join(args.log_dir, 'aggregator_ckpt.pth')
        start_epoch = 0
        best_loss = float('inf')
        best_state = None
        not_improved_count = 0

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            aggregator.load_state_dict(ckpt['aggregator_state_dict'])
            agg_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            best_loss = ckpt.get('best_loss', float('inf'))
            not_improved_count = ckpt.get('not_improved_count', 0)
            if ckpt.get('best_state') is not None:
                best_state = ckpt['best_state']
            self.logger.info(f'[Aggregator] Resumed from checkpoint: epoch {start_epoch}, '
                             f'best_val_loss={best_loss:.6f}')

        # Determine dataset name for scaler
        select_dataset = args.dataset_use[0]

        # 5. Pre-compute base model outputs into CPU RAM (one-time cost)
        self.logger.info('[Aggregator] Phase 1: pre-computing base model outputs into RAM ...')
        train_cache = self._precompute_inmemory(self.train_dataloader, desc='Train cache')
        val_cache = None
        if self.val_dataloader is not None:
            val_cache = self._precompute_inmemory(self.val_dataloader, desc='Val cache')
        self.logger.info('[Aggregator] Phase 1 complete. Starting Phase 2 (aggregator training) ...')

        # 6. Training loop from cached data
        for epoch in range(start_epoch, args.aggregator_epochs):
            aggregator.train()
            total_loss = 0
            step = 0

            # Shuffle batch order each epoch for better training
            indices = list(range(len(train_cache)))
            _random.shuffle(indices)

            for i, cache_idx in enumerate(indices):
                cached = train_cache[cache_idx]
                query_pred  = cached['query_pred'].float().to(args.device)
                query_enc   = cached['query_enc'].float().to(args.device)
                corrections = cached['corrections'].float().to(args.device)
                demo_encs   = cached['demo_encs'].float().to(args.device)
                tgt         = cached['targets'].float().to(args.device)

                # --- Aggregator forward + backward ---
                weighted_correction = aggregator(query_enc, demo_encs, corrections)
                output = query_pred + weighted_correction

                agg_optimizer.zero_grad()
                loss = self.loss(output, tgt, self.scaler_dict[select_dataset])
                loss.backward()

                if args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(aggregator.parameters(), args.max_grad_norm)
                agg_optimizer.step()

                total_loss += loss.item()
                step += 1
                self.logger.info(
                    f'[Aggregator] epoch {epoch} batch {i}/{len(indices)}, '
                    f'batch_loss={loss.item():.6f}, avg_loss={total_loss / step:.6f}'
                )

            train_loss = total_loss / max(step, 1)
            self.logger.info(f'[Aggregator] Epoch {epoch}: train_loss={train_loss:.6f}')

            # --- Validation from cache ---
            if val_cache is not None:
                val_loss = self._val_aggregator_cached(aggregator, val_cache, select_dataset)
                self.logger.info(f'[Aggregator] Epoch {epoch}: val_loss={val_loss:.6f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    not_improved_count = 0
                    best_state = copy.deepcopy(aggregator.state_dict())
                    self.logger.info('[Aggregator] **** New best model saved!')
                else:
                    not_improved_count += 1

                if args.early_stop and not_improved_count >= args.early_stop_patience:
                    self.logger.info(f'[Aggregator] Early stopping after {epoch + 1} epochs')
                    break
            else:
                best_state = copy.deepcopy(aggregator.state_dict())

            # Save checkpoint at end of each epoch
            torch.save({
                'aggregator_state_dict': aggregator.state_dict(),
                'optimizer_state_dict': agg_optimizer.state_dict(),
                'epoch': epoch + 1,
                'batch_idx': 0,
                'best_loss': best_loss,
                'best_state': best_state,
                'not_improved_count': not_improved_count,
            }, ckpt_path)

        # Save best aggregator weights
        if best_state is not None:
            save_path = os.path.join(args.log_dir, 'aggregator_best.pth')
            torch.save(best_state, save_path)
            self.logger.info(f'[Aggregator] Best weights saved to {save_path}')
            aggregator.load_state_dict(best_state)

    def _val_aggregator_cached(self, aggregator, val_cache, select_dataset):
        """Validation using pre-cached base model outputs from RAM."""
        aggregator.eval()
        args = self.args
        total_val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for cached in val_cache:
                query_pred  = cached['query_pred'].float().to(args.device)
                query_enc   = cached['query_enc'].float().to(args.device)
                corrections = cached['corrections'].float().to(args.device)
                demo_encs   = cached['demo_encs'].float().to(args.device)
                tgt         = cached['targets'].float().to(args.device)

                weighted_correction = aggregator(query_enc, demo_encs, corrections)
                output = query_pred + weighted_correction

                loss = self.loss(output, tgt, self.scaler_dict[select_dataset])
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                n_batches += 1

        return total_val_loss / max(n_batches, 1)

    @staticmethod
    def test_ict(model, args, scaler_dict, test_dataloader, logger, path=None,
                 num_prefix_selections=1, ict_mode='residual'):
        """
        ICT inference: pure forward pass with demonstrations, no gradient updates.

        Args:
            num_prefix_selections: number of random demo sets (S) to average over
                                   (reduces variance from demo selection)
        """
        # Load pretrained weights
        if path is not None:
            if str(args.device) != 'cpu' and torch.cuda.device_count() > 1:
                model.load_state_dict(torch.load(path, map_location=args.device))
            else:
                model_weights = {k.replace('module.', ''): v for k, v in torch.load(path, map_location=args.device).items()}
                model.load_state_dict(model_weights)
            model.to(args.device)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_samples = 0
            batch_idx = 0
            num_batches = len(test_dataloader)

            for batch_data in test_dataloader:
                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(args.device)      # [B, T, N, F]
                targets = targets.squeeze(0).to(args.device)     # [B, T, N, F]
                demos_x = demos_x.squeeze(0).to(args.device)    # [B, S, K, T, N, F]
                demos_y = demos_y.squeeze(0).to(args.device)    # [B, S, K, T, N, F]

                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                S = demos_x.shape[1]

                # Average predictions over S independent demo selections
                outputs = []
                for s in range(S):
                    output_s = model(inputs, targets, select_dataset,
                                     demos_x=demos_x[:, s],    # [B, K, T, N, F]
                                     demos_y=demos_y[:, s],     # [B, K, T, N, F]
                                     ict_mode=ict_mode)
                    outputs.append(output_s)
                output = torch.stack(outputs).mean(dim=0)        # [B, T, N, 1]

                if args.real_value == False:
                    output = scaler_dict[select_dataset].inverse_transform(output)
                    y_lbl = scaler_dict[select_dataset].inverse_transform(targets[..., :args.output_dim])
                else:
                    y_lbl = targets[..., :args.output_dim]

                batch_mae, batch_rmse, batch_mape, batch_mse, corr, mae_count, rmse_count, mse_count, mape_count = \
                    All_Metrics(output, y_lbl, args.mae_thresh, args.mape_thresh)
                mae += batch_mae * mae_count
                rmse += batch_mse * rmse_count
                mape += batch_mape * mape_count
                total_count += mae_count
                total_mape_count += mape_count
                total_samples += len(y_lbl)
                batch_idx += 1
                logger.info(
                    f'[ICT] batch {batch_idx}/{num_batches}, samples={total_samples}, '
                    f'MAE={batch_mae:.4f}, RMSE={batch_rmse:.4f}, MAPE={batch_mape:.4f}, '
                    f'mae_count={mae_count}, mape_count={mape_count}'
                )

                # Free GPU memory between batches
                del demos_x, demos_y, outputs
                if torch.cuda.is_available() and str(args.device) != 'cpu':
                    torch.cuda.empty_cache()

        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count

        logger.info(f'[ICT] === Test Summary ===')
        logger.info(f'[ICT] total_batches={batch_idx}, total_mae_count={total_count}, total_mape_count={total_mape_count}')
        logger.info("[ICT] MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
            mae, rmse, mape * 100, corr))
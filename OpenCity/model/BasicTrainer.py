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
        if os.path.isdir(args.log_dir) == False and not args.debug:
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
        print("Pre-train finish.")


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
                if args.model == 'OpenCity':
                    print(total_batch, batch_mae, batch_rmse, batch_mape, total_count, total_mape_count)
        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count
        print('last batch', output.shape, y_lbl.shape)
        print(total_batch, total_count, total_mape_count)

        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR:{:.4f}".format(
            mae, rmse, mape * 100, corr))

    def train_aggregator(self):
        """Train only the DemoAggregator while keeping the base model frozen.

        Uses ICT dataloaders (4-tuple: inputs, targets, demos_x, demos_y).
        Only aggregator parameters receive gradients.
        """
        args = self.args

        # 1. Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

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

        best_loss = float('inf')
        best_state = None
        not_improved_count = 0

        for epoch in tqdm(range(args.aggregator_epochs)):
            # --- Train ---
            self.model.train()
            total_loss = 0
            step = 0

            for batch_data in self.train_dataloader:
                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(args.device)
                targets = targets.squeeze(0).to(args.device)
                demos_x = demos_x.squeeze(0).to(args.device)  # [B, S, K, T, N, F]
                demos_y = demos_y.squeeze(0).to(args.device)

                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])

                # Use first demo selection (S=0) for training
                output = self.model(inputs, targets, select_dataset,
                                    demos_x=demos_x[:, 0],  # [B, K, T, N, F]
                                    demos_y=demos_y[:, 0],
                                    ict_mode='learned')

                agg_optimizer.zero_grad()
                loss = self.loss(output, targets[..., :args.output_dim],
                                 self.scaler_dict[select_dataset])
                loss.backward()

                if args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(aggregator.parameters(), args.max_grad_norm)
                agg_optimizer.step()

                total_loss += loss.item()
                step += 1
                if step % args.log_step == 0:
                    self.logger.info(f'[Aggregator] epoch {epoch} step {step} '
                                     f'train_loss={total_loss / step:.6f}')

            train_loss = total_loss / max(len(self.train_dataloader), 1)
            self.logger.info(f'[Aggregator] Epoch {epoch}: train_loss={train_loss:.6f}')

            # --- Validation ---
            if self.val_dataloader is not None:
                val_loss = self._val_aggregator_epoch()
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
                # No validation: always save latest
                best_state = copy.deepcopy(aggregator.state_dict())

        # Save best aggregator weights
        if best_state is not None:
            save_path = os.path.join(args.log_dir, 'aggregator_best.pth')
            torch.save(best_state, save_path)
            self.logger.info(f'[Aggregator] Best weights saved to {save_path}')
            aggregator.load_state_dict(best_state)

    def _val_aggregator_epoch(self):
        """Validation epoch for aggregator training."""
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in self.val_dataloader:
                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(self.args.device)
                targets = targets.squeeze(0).to(self.args.device)
                demos_x = demos_x.squeeze(0).to(self.args.device)
                demos_y = demos_y.squeeze(0).to(self.args.device)

                select_dataset = get_key_from_value(self.num_nodes_dict, inputs.shape[2])

                output = self.model(inputs, targets, select_dataset,
                                    demos_x=demos_x[:, 0],
                                    demos_y=demos_y[:, 0],
                                    ict_mode='learned')

                loss = self.loss(output, targets[..., :self.args.output_dim],
                                 self.scaler_dict[select_dataset])
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        return total_val_loss / max(len(self.val_dataloader), 1)

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

        # Determine autocast dtype for memory optimization
        # Use bfloat16: same dynamic range as float32, avoids NaN overflow
        use_amp = torch.cuda.is_available() and str(args.device) != 'cpu'
        amp_dtype = torch.bfloat16

        with torch.no_grad():
            mae = 0
            rmse = 0
            mape = 0
            total_count = 0
            total_mape_count = 0
            total_batch = 0

            for batch_data in test_dataloader:
                inputs, targets, demos_x, demos_y = batch_data
                inputs = inputs.squeeze(0).to(args.device)      # [B, T, N, F]
                targets = targets.squeeze(0).to(args.device)     # [B, T, N, F]
                demos_x = demos_x.squeeze(0).to(args.device)    # [B, S, K, T, N, F]
                demos_y = demos_y.squeeze(0).to(args.device)    # [B, S, K, T, N, F]

                # Convert inputs to model's dtype (float16 if model.half() was called)
                model_dtype = next(model.parameters()).dtype
                inputs_m = inputs.to(model_dtype)
                targets_m = targets.to(model_dtype)
                demos_x_m = demos_x.to(model_dtype)
                demos_y_m = demos_y.to(model_dtype)

                select_dataset = get_key_from_value(args.num_nodes_dict, inputs.shape[2])
                S = demos_x_m.shape[1]

                # Average predictions over S independent demo selections
                outputs = []
                for s in range(S):
                    with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=amp_dtype, enabled=use_amp):
                        output_s = model(inputs_m, targets_m, select_dataset,
                                         demos_x=demos_x_m[:, s],    # [B, K, T, N, F]
                                         demos_y=demos_y_m[:, s],     # [B, K, T, N, F]
                                         ict_mode=ict_mode)
                    outputs.append(output_s.float())  # cast back to float32 for metrics
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
                total_batch += len(y_lbl)
                print(f'[ICT] batch {total_batch}, MAE: {batch_mae:.4f}, RMSE: {batch_rmse:.4f}, MAPE: {batch_mape:.4f}')

                # Free GPU memory between batches
                del demos_x, demos_y, demos_x_m, demos_y_m, outputs, inputs_m, targets_m
                if torch.cuda.is_available() and str(args.device) != 'cpu':
                    torch.cuda.empty_cache()

        mae /= total_count
        rmse = (rmse / total_count) ** 0.5
        mape /= total_mape_count
        print('last batch', output.shape, y_lbl.shape)
        print(total_batch, total_count, total_mape_count)

        logger.info("ICT Test — MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, CORR: {:.4f}".format(
            mae, rmse, mape * 100, corr))
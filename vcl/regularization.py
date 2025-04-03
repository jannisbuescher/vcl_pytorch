import torch
import torch.nn as nn
import random


class L2(nn.Module):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, agent_config):
        super(L2, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.online_reg = True
        self.damping_factor = 0.1
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance
    

    def learn_batch(self, train_loader, val_loader=None):

        self.log('#reg_term:', len(self.regularization_terms))

        # 1.Learn the parameters for current task
        super(L2, self).learn_batch(train_loader, val_loader)

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}


    def criterion(self, inputs, targets, tasks, regularization=True, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t,t_preds in inputs.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = inputs['All']
            if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = inputs['All'][:,:self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)

        if regularization and len(self.regularization_terms)>0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.config['reg_coef'] * reg_loss
        return loss
    

    # def update_model(self, inputs, targets, tasks):

    #     unreg_gradients = {}
        
    #     # 1.Save current parameters
    #     old_params = {}
    #     for n, p in self.params.items():
    #         old_params[n] = p.clone().detach()

    #     # 2. Collect the gradients without regularization term
    #     out = self.forward(inputs)
    #     loss = self.criterion(out, targets, tasks, regularization=False)
    #     self.optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     for n, p in self.params.items():
    #         if p.grad is not None:
    #             unreg_gradients[n] = p.grad.clone().detach()

    #     # 3. Normal update with regularization
    #     loss = self.criterion(out, targets, tasks, regularization=True)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # 4. Accumulate the w
    #     for n, p in self.params.items():
    #         delta = p.detach() - old_params[n]
    #         if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
    #             self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

    #     return loss.detach(), out


    # - Alternative simplified implementation with similar performance -
    def update_model(self, inputs, targets, tasks):
        # A wrapper of original update step to include the estimation of w

        # Backup prev param if not done yet
        # The backup only happened at the beginning of a new task
        if len(self.prev_params) == 0:
            for n, p in self.params.items():
                self.prev_params[n] = p.clone().detach()

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2.Calculate the loss as usual
        loss, out = super(SI, self).update_model(inputs, targets, tasks)

        # 3.Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if p.grad is not None:
                self.w[n] -= p.grad * delta

        return loss.detach(), out


    # def learn_batch(self, train_loader, val_loader=None):
    #     if self.reset_optimizer:  # Reset optimizer before learning each task
    #         self.log('Optimizer is reset!')
    #         self.init_optimizer()

    #     for epoch in range(self.config['schedule'][-1]):
    #         data_timer = Timer()
    #         batch_timer = Timer()
    #         batch_time = AverageMeter()
    #         data_time = AverageMeter()
    #         losses = AverageMeter()
    #         acc = AverageMeter()

    #         # Config the model and optimizer
    #         self.log('Epoch:{0}'.format(epoch))
    #         self.model.train()
    #         self.scheduler.step(epoch)
    #         for param_group in self.optimizer.param_groups:
    #             self.log('LR:',param_group['lr'])

    #         # Learning with mini-batch
    #         data_timer.tic()
    #         batch_timer.tic()
    #         self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
    #         for i, (input, target, task) in enumerate(train_loader):

    #             data_time.update(data_timer.toc())  # measure data loading time

    #             if self.gpu:
    #                 input = input.cuda()
    #                 target = target.cuda()

    #             loss, output = self.update_model(input, target, task)
    #             input = input.detach()
    #             target = target.detach()

    #             # measure accuracy and record loss
    #             acc = accumulate_acc(output, target, task, acc)
    #             losses.update(loss, input.size(0))

    #             batch_time.update(batch_timer.toc())  # measure elapsed time
    #             data_timer.toc()

    #             if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
    #                 self.log('[{0}/{1}]\t'
    #                       '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
    #                       '{data_time.val:.4f} ({data_time.avg:.4f})\t'
    #                       '{loss.val:.3f} ({loss.avg:.3f})\t'
    #                       '{acc.val:.2f} ({acc.avg:.2f})'.format(
    #                     i, len(train_loader), batch_time=batch_time,
    #                     data_time=data_time, loss=losses, acc=acc))

    #         self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

    #         # Evaluate the performance of current task
    #         if val_loader != None:
    #             self.validation(val_loader)

class SI(L2):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """

    def __init__(self, agent_config):
        super(SI, self).__init__(agent_config)
        self.online_reg = True  # Original SI works in an online updating fashion
        self.damping_factor = 0.1
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def update_model(self, inputs, targets, tasks):

        unreg_gradients = {}
        
        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss.detach(), out

    """
    # - Alternative simplified implementation with similar performance -
    def update_model(self, inputs, targets, tasks):
        # A wrapper of original update step to include the estimation of w

        # Backup prev param if not done yet
        # The backup only happened at the beginning of a new task
        if len(self.prev_params) == 0:
            for n, p in self.params.items():
                self.prev_params[n] = p.clone().detach()

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2.Calculate the loss as usual
        loss, out = super(SI, self).update_model(inputs, targets, tasks)

        # 3.Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if p.grad is not None:  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= p.grad * delta  # w[n] is >=0

        return loss.detach(), out
    """

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg,'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance


class MAS(L2):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, agent_config):
        super(MAS, self).__init__(agent_config)
        self.online_reg = True

    def calculate_importance(self, dataloader):
        self.log('Computing MAS')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the gradients of L2 loss on the outputs
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the  loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,:self.valid_out_dim]

            pred.pow_(2)
            loss = pred.mean()

            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += (self.params[n].grad.abs() / len(dataloader))

        self.train(mode=mode)

        return importance
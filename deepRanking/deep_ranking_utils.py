from itertools import combinations
from datetime import date

import torchvision.utils
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import numpy as np
import torch
import datetime

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile # to fix a bug:

cuda = torch.cuda.is_available()


# utility functions
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class Experiment:
    def __init__(self, train_loader, val_loader, model, loss_fn, optimizer, scheduler, cuda, log_interval=50,
                 to_tensorboard=True, metrics=[], start_epoch=0, margin=1, lr=0.01, n_epochs=10):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.cuda = cuda
        self.log_interval = log_interval
        self.to_tensorboard = to_tensorboard
        self.metrics = metrics
        self.start_epoch = start_epoch
        self.margin = margin
        self.lr = lr
        self.step = 0
        self.val_loss = 0
        self.train_loss = 0

        if self.to_tensorboard:
            now = datetime.datetime.now()
            self.writer = SummaryWriter(
                    f'runs/{date.today().strftime("%b-%d-%Y")}/{self.n_epochs}ep_{self.margin}m_{self.lr}lr_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.train_loss, self.val_loss = self.fit()
            self.writer.close()

        else:
            self.train_loss, self.val_loss = self.fit()

    def fit(self):
        val_losses = []
        training_losses = []

        for epoch in range(0, self.start_epoch):
            self.scheduler.step()

        for epoch in range(self.start_epoch, self.n_epochs):
            self.scheduler.step()

            train_loss, metrics = self.train_epoch()
            training_losses += [train_loss]

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, self.n_epochs, train_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss, metrics = self.test_epoch()
            val_loss /= len(self.val_loader)
            val_losses += [val_loss]

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, self.n_epochs,
                                                                                     val_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)

            if self.to_tensorboard:
                self.writer.add_hparams({'n_epochs': self.n_epochs, 'lr': self.lr, 'margin': self.margin},
                                        {'Avr. Training Loss': sum(training_losses) / len(training_losses)})

        return sum(training_losses) / len(training_losses), sum(val_losses) / len(val_losses)

    def train_epoch(self):
        for metric in self.metrics:
            metric.reset()

        self.model.train()
        losses = []
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            print('Batch [{}/{}]'.format(batch_idx, len(self.train_loader)))
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if self.cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = self.loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            triplet_ids = loss_outputs[2]
            self.optimizer.step()

            if self.to_tensorboard:
                if batch_idx == len(self.train_loader) - 1:
                    # create image grid of input images from the batch
                    img_grid = self.make_triplet_grid(triplet_ids, data)
                    self.writer.add_image("Training input triplets", img_grid, global_step=batch_idx)


                # add the weights from the last layer as a histogram:
                self.writer.add_histogram("Weights from the last linear layer", self.model.fc[4].weight, global_step=self.step)
                # add the training loss for the specific batch:
                self.writer.add_scalar("Training loss", loss, global_step=self.step) # should be running loss or not?

            for metric in self.metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % self.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), np.mean(losses))
                for metric in self.metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

            self.step += 1

        total_loss /= (batch_idx + 1)
        return total_loss, self.metrics

    def test_epoch(self):
        # self.step = 0  # reset step (is used for plotting in tensorboard)?? should we do this?
        with torch.no_grad():
            for metric in self.metrics:
                metric.reset()
            self.model.eval()
            val_loss = 0
            for batch_idx, (data, target) in enumerate(self.val_loader):
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                if cuda:
                    data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()

                outputs = self.model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = self.loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()

                for metric in self.metrics:
                    metric(outputs, target, loss_outputs)

                if self.to_tensorboard:
                    # make 3d plot of embeddings
                    features = loss_inputs[0]  # the embeddings
                    labels = loss_inputs[1].tolist()  # the product ids
                    label_img = data[0] # the original images
                    self.writer.add_embedding(features, metadata=labels, label_img=label_img, global_step=batch_idx)

                self.step += 1

        return val_loss, self.metrics

    def make_triplet_grid(self, triplet_idxs, data):
        # should make a grid of the different triplets used
        triplet_grids = []
        for triplet in triplet_idxs:
            triplet_grids += [torchvision.utils.make_grid([data[0][triplet[0]], data[0][triplet[1]], data[0][triplet[2]]])]

        final_grid = torchvision.utils.make_grid(triplet_grids)
        return final_grid


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                               negative_selection_fn=lambda
                                                                                                   x: semihard_negative(
                                                                                                   x, margin),
                                                                                               cpu=cpu)

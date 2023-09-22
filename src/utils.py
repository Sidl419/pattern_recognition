import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import Scaler

import torch
from sklearn.metrics import f1_score  
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from statsmodels.stats.proportion import proportion_confint 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class P300Getter:
    def __init__(self, raw_data, eloc, n_channels=64, sfreq=120, sample_size=72, target_chars=None):
        self.raw_data = raw_data
        self.target_chars = target_chars
        self.sample_size = sample_size
        self.n_channels = n_channels

        self._X = None
        self._y = None

        self.eloc = eloc
        self.info = mne.create_info(ch_names=eloc.ch_names, ch_types=['eeg'] * n_channels, sfreq=sfreq)
        self.scaler = Scaler(self.info)

        self.row_dict = []
        self.row_dict += list(zip(['A', 'B', 'C', 'D', 'E', 'F'], [7] * 6))
        self.row_dict += list(zip(['G', 'H', 'I', 'J', 'K', 'L'], [8] * 6))
        self.row_dict += list(zip(['M', 'N', 'O', 'P', 'Q', 'R'], [9] * 6))
        self.row_dict += list(zip(['S', 'T', 'U', 'V', 'W', 'X'], [10] * 6))
        self.row_dict += list(zip(['Y', 'Z', '1', '2', '3', '4'], [11] * 6))
        self.row_dict += list(zip(['5', '6', '7', '8', '9', '_'], [12] * 6))
        self.row_dict = dict(self.row_dict)

        self.col_dict = []
        self.col_dict += list(zip(['A', 'G', 'M', 'S', 'Y', '5'], [1] * 6))
        self.col_dict += list(zip(['B', 'H', 'N', 'T', 'Z', '6'], [2] * 6))
        self.col_dict += list(zip(['C', 'I', 'O', 'U', '1', '7'], [3] * 6))
        self.col_dict += list(zip(['D', 'J', 'P', 'V', '2', '8'], [4] * 6))
        self.col_dict += list(zip(['E', 'K', 'Q', 'W', '3', '9'], [5] * 6))
        self.col_dict += list(zip(['F', 'L', 'R', 'X', '4', '_'], [6] * 6))
        self.col_dict = dict(self.col_dict)
    
    def filter(self, X, freq=120):
        train_array = mne.io.RawArray(X.T, self.info, verbose=False)
        train_array.set_montage(self.eloc)
        self.info = mne.create_info(ch_names=self.eloc.ch_names, ch_types=['eeg'] * self.n_channels, sfreq=freq)
        return train_array.filter(1, 20, method='fir', verbose=False).resample(freq, verbose=False).get_data()

    def to_tensor(self, X_train, y_train):
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

        return X_train.float(), y_train.float()

    def upsample(self, pos_rate=4):
        pos_idx = np.where(self._y == 1)[0]
        pos_X = self._X[pos_idx]
        pos_y = np.ones(len(pos_idx))

        for _ in range(pos_rate):
            self._X = np.concatenate([self._X, pos_X])
            self._y = np.concatenate([self._y, pos_y])

    def get_cnn_p300_dataset(self, filter=False):
        X = []
        y = []

        for epoch_num in tqdm(range(len(self.raw_data['Flashing']))):
            epoch_flash = self.raw_data['Flashing'][epoch_num]
            idx = np.where(epoch_flash[:-1] != epoch_flash[1:])[0][1::2] + 1
            idx = np.concatenate([[0], idx])
            
            if filter:
                data = self.filter(self.raw_data['Signal'][epoch_num])
            else:
                data = self.raw_data['Signal'][epoch_num].T

            res = []
            bias = 24
            for i in idx:
                res.append(data[:, i+bias:i+self.sample_size])

            if self.target_chars:
                rows = (self.raw_data['StimulusCode'][epoch_num][idx] == self.row_dict[self.target_chars[epoch_num]]).astype(int)
                cols = (self.raw_data['StimulusCode'][epoch_num][idx] == self.col_dict[self.target_chars[epoch_num]]).astype(int)

                target = rows + cols
            else:
                target = self.raw_data['StimulusType'][epoch_num][idx]

            X.append(res)
            y.append(target)

        self._X = np.concatenate(X)
        self._y = np.concatenate(y)

    def get_data(self):
        data = self.scaler.fit_transform(self._X)
        data, target = self.to_tensor(data, self._y)

        return data, target

    def unscale(self, X):
        return self.scaler.inverse_transform(X)


def get_motor_subject(subject = 1):
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs, verbose=False, update_path=True)
    raw = concatenate_raws([read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames], verbose=False)
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                      exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, verbose=False)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    return epochs_train.get_data(), labels

def to_tensor(X_train, y_train):
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    return X_train.float(), y_train.float()

def get_cursor_data(info):
    X = []
    y = []
    for _ in tqdm(range(1, 110)):
        t_X, t_y = get_motor_subject()
        X.append(t_X)
        y.append(t_y)

    X = np.concatenate(X)
    y = np.concatenate(y)
    X, y = to_tensor(X, y)

    scaler = Scaler(info)

    X = scaler.fit_transform(X)

    return X, y


def train_model(model, dataloaders, criterion, learning_params, device='cpu', log_rate=10):
    since = time.time()

    optimizer = optim.AdamW(model.parameters(), lr=learning_params['lr'], weight_decay=learning_params['weight_decay'])
    scheduler = StepLR(optimizer, step_size=learning_params['step_size'], gamma=learning_params['gamma'])
    model = model.to(device)

    val_acc_history, val_loss_history, val_f1_history, val_bc_history = [], [], [], []

    for epoch in range(learning_params['num_epochs']):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_ones = 0
            running_TP, running_TN, running_FP, running_FN = 0, 0, 0, 0

            for data in dataloaders[phase]:
                if learning_params['model_type'] == 'GNN':
                    inputs = data.to(device)
                    labels = data.y.to(device)
                    inputs_size = inputs.x.size(0)
                elif learning_params['model_type'] == 'CNN':
                    inputs = data[0].to(device)
                    labels = data[1].to(device)
                    inputs_size = inputs.size(0)
                else:
                    raise ValueError(f"no such model type: {learning_params['model_type']}")

                optimizer.zero_grad()

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                _, true_y = torch.max(labels.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                P = torch.sum(preds)
                N = torch.sum(1 - preds)
                TP = torch.sum(torch.masked_select(true_y, preds == 1))
                TN = torch.sum(torch.masked_select(1 - true_y, preds == 0))
                FP = P - TP
                FN = N - TN
                
                running_loss += loss.item() * inputs_size
                running_corrects += torch.sum(preds == true_y)
                running_ones += P
                running_TP += TP
                running_TN += TN
                running_FP += FP
                running_FN += FN

            epoch_loss = running_loss / len(dataloaders[phase].dataset) 
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_ones = running_ones.double() / (len(dataloaders[phase].dataset)  // dataloaders[phase].batch_size)
            epoch_precision = running_TP.double() / (running_TP + running_FP)
            epoch_recall = running_TP.double() / (running_TP + running_FN)
            epoch_f1 = 2 * (epoch_precision * epoch_recall) / (epoch_precision + epoch_recall)
            epoch_bc = (epoch_recall + running_TN.double() / (running_TN + running_FP)) / 2

            min_acc, max_acc = proportion_confint(running_corrects.cpu(), len(dataloaders[phase].dataset))

            if (epoch + 1) % log_rate == 0:
                if phase == 'train':
                    print('Epoch {}/{}'.format(epoch, learning_params['num_epochs'] - 1))
                    print('-' * 150)
                print('{}\t Loss: {:.4f}\t Min Acc: {:.4f}\t Acc: {:.4f}\t Max Acc: {:.4f}\t Balanced Acc: {:.4f}\t Positive: {:.4f}\t Precision: {:.4f}\t Recall: {:.4f}\t F1-score: {:.4f}\t'.format(phase, 
                        epoch_loss, min_acc, epoch_acc, max_acc, epoch_bc, epoch_ones, epoch_precision, epoch_recall, epoch_f1))

            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu().data)
                val_loss_history.append(epoch_loss)
                val_f1_history.append(epoch_f1.cpu())
                val_bc_history.append(epoch_bc.cpu())

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    acc = {'Accuracy': np.array(val_acc_history), 
           'Balanced Accuracy': np.array(val_bc_history), 
           'F1-score': np.array(val_f1_history)}

    return np.array(val_loss_history), acc, time_elapsed


def plot_sample(raw_dataset, signal_sample, info, is_mean=False):
    output = raw_dataset.unscale(signal_sample.numpy())[0]

    plt.figure(figsize=(10, 10))
    mean_output = output.mean(axis=0)
    t_axis = np.arange(len(mean_output)) / info['sfreq'] * 1000
    plt.plot(t_axis, mean_output)
    plt.ylabel('amplitude (muV)')
    plt.xlabel('time (ms)')
    plt.title('Averaged EEG signal')
    plt.show()

    mne_output = mne.io.RawArray(output, info=info, verbose=False)
    plt.figure(figsize=(10, 10))
    mne_output.plot(
                n_channels=len(info['ch_names']), scalings='auto', 
                title='Raw EEG signal')
    plt.show()


def show_progress(loss, metric, loss_title, metric_title):
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(15)

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(loss)), loss, 'r', linewidth=2)

    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('mean loss', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(loss_title, fontsize=14)
    
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(metric[metric_title])), metric[metric_title], 'b', linewidth=2)

    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('criterion', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(metric_title, fontsize=14)

    plt.grid()
    plt.show()

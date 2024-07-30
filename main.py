import numpy as np
import torch
import torchaudio.functional
from torch import nn
from torchaudio.datasets import SPEECHCOMMANDS
import os
from functools import partial
from torch.utils.data import DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from torchinfo import summary
import torchaudio

# Set the backend to 'sox_io' or 'soundfile'
torchaudio.set_audio_backend("sox_io")
# or
# torchaudio.set_audio_backend("soundfile")
matplotlib.use('Agg')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


###########################################################
# Google Speech Command DataSet
###########################################################
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data.
train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

# check the partition between the datasets:
print(f"\nTrain set: {len(train_set)} | {100 * len(train_set) / (len(train_set) + len(val_set) + len(test_set)):.1f}%")
print(f"Validation set: {len(val_set)} | {100 * len(val_set) / (len(train_set) + len(val_set) + len(test_set)):.1f}%")
print(f"Test set: {len(test_set)} | {100 * len(test_set) / (len(train_set) + len(val_set) + len(test_set)):.1f}%\n")

# each "sample" is composed of the tuple: (waveform, sample_rate, label, speaker_id, utterance_number)
# waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels = set()
for fp in Path('./SpeechCommands/speech_commands_v0.02/').glob("**/*.wav"):
    labels.add(fp.parent.name)
labels = sorted(labels)
labels.remove('_background_noise_')
print(labels)
NUM_CLASSES = len(labels)
print(f"#Labels: {len(labels)}")
L2I = {l: i for i, l in enumerate(labels)}
I2L = {i: l for i, l in enumerate(labels)}
for l, i in L2I.items():
    print(f"{l} -> {i}")


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        waveform = torch.flatten(waveform)
        # make sure we have exactly 1sec (i.e., 16000)
        if len(waveform) < 16_000:
            waveform = torch.nn.functional.pad(waveform, (0, 16_000 - len(waveform)))
        elif len(waveform) > 16_000:
            waveform = waveform[:16_000]

        tensors += [waveform]
        targets += [L2I[label]]

    # Group the list of tensors into a batched tensor
    tensors = torch.stack(tensors, dim=0)
    targets = torch.tensor(targets)

    return tensors, targets


###########################################################
#                       Model
###########################################################
class DeepGSC(nn.Module):
    def __init__(self, num_classes, sample_rate=16_000, window_len=0.032, hop_size=0.01, nfft=512, num_mel=32):
        super().__init__()

        self.num_classes = num_classes

        self.sr = sample_rate
        self.nfft = nfft
        self.num_mel = num_mel
        self.win_len = int(sample_rate * window_len)  # sec ==> samples
        self.hop_size = int(sample_rate * hop_size)  # sec ==> samples
        self.window = torch.hann_window(self.win_len)

        # init our stft function
        self.stft = partial(
            torch.stft,
            n_fft=nfft,
            hop_length=self.hop_size,
            win_length=self.win_len,
            return_complex=True,
            center=False
        )
        self.mel_filters = torchaudio.functional.melscale_fbanks(
            int(nfft // 2 + 1),
            n_mels=num_mel,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            sample_rate=sample_rate,
        )

        # init the model itself:
        self.H = num_mel
        self.W = int(((sample_rate - (self.win_len - 1) - 1) / self.hop_size) + 1)
        C = 1

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=16, kernel_size=5, padding='same'),  # (B, C, H, W) --> (B, 16, H, W)
            nn.BatchNorm2d(num_features=16),  # (B, 16, H, W) --> (B, 16, H, W)
            nn.ReLU(),  # (B, 16, H, W) --> (B, 16, H, W)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (B, 16, H, W) --> (B, 16, H, W/2)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding='same'),
            # (B, 16, H, W/2) --> (B, 32, H, W/2)
            nn.BatchNorm2d(num_features=32),  # (B, 32, H, W/2) --> (B, 32, H, W/2)
            nn.ReLU(),  # (B, 32, H, W/2) --> (B, 32, H, W/2)
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # (B, 32, H, W/2) --> (B, 32, H, W/4)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            # (B, 32, H, W/4) --> (B, 64, H, W/4)
            nn.BatchNorm2d(num_features=64),  # (B, 64, H, W/4) --> (B, 64, H, W/4)
            nn.ReLU(),  # (B, 64, H, W/4) --> (B, 64, H, W/4)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (B, 64, H, W/4) --> (B, 64, H/2, W/8)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            # (B, 64, H/2, W/8) --> (B, 128, H/2, W/8)
            nn.BatchNorm2d(num_features=128),  # (B, 128, H/2, W/8) --> (B, 128, H/2, W/8)
            nn.ReLU(),  # (B, 128, H/2, W/8) --> (B, 128, H/2, W/8)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (B, 128, H/2, W/8) --> (B, 128, H/4, W/16)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            # (B, 128, H/4, W/16) --> (B, 256, H/4, W/16)
            nn.BatchNorm2d(num_features=256),  # (B, 256, H/4, W/16) --> (B, 256, H/4, W/16)
            nn.ReLU(),  # (B, 256, H/4, W/16) --> (B, 256, H/4, W/16)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (B, 256, H/4, W/16) --> (B, 256, H/8, W/32)

            nn.Flatten(),  # (B, 256, H/8, W/32) --> (B, 256*(H/8)*(W/32))

            nn.Linear(in_features=256 * (self.H // 8) * (self.W // 32), out_features=128),
            # (B, 256*(H/8)*(W/32)) --> (B, 128)
            nn.Dropout(p=0.25),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),  # (B, 128) --> (B, 128)
            nn.Dropout(p=0.25),
            nn.ReLU(),

            # projection (final) layer
            nn.Linear(in_features=128, out_features=self.num_classes),  # (B, 128) --> (B, #classes)
            nn.Softmax(dim=-1)
        )

        self.print_model_weights()
        print(summary(self, (1, 16000)))

    def feature_extraction(self, waveform):
        # input shape: (B, 16000)
        self.window = self.window.to(waveform.device)
        self.mel_filters = self.mel_filters.to(waveform.device)
        spectrogram = self.stft(waveform, window=self.window)  # (B, 16000) --> (B, self.nfft/2 + 1, W)
        power = spectrogram.abs() ** 2
        mel = self.mel_filters.t() @ power
        logmel = torch.log10(mel + 1e-10)
        normalized_logmel = (logmel - logmel.mean(dim=(1, 2), keepdims=True)) / (
                    logmel.std(dim=(1, 2), keepdims=True) + 1e-10)
        return normalized_logmel

    def forward(self, waveform):
        # waveform.shape = (B, 16_000)
        normalized_logmel = self.feature_extraction(waveform)  # (B, 16000) --> (B, H, W)

        # add input channels:
        x = normalized_logmel.unsqueeze(dim=1)  # (B, H, W) --> (B, C=1, H, W)

        # got through the model's layers:
        prob = self.layers(x)  # (B, C, H, W) --> (B, #classes)
        return prob

    def print_model_weights(self):
        """
        Prints the number of weights in the PyTorch model and its layers.

        Args:
            model (torch.nn.Module): The PyTorch model.
        """
        print(f"Model: {self.__class__.__name__}")
        total_weights = 0
        for name, param in self.named_parameters():
            num_weights = param.numel()
            total_weights += num_weights
            print(f"Layer: {name} | Number of weights: {num_weights:,}")
        print(f"Total number of weights: {total_weights:,}")


###########################################################
#                    Training Loop
###########################################################
class Trainer:
    def __init__(self, model: torch.nn.Module, checkpoint, batch_size: int, learning_rate: float, num_epochs: int):
        self.training_dataloader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        self.test_dataloader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn
        )

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            self.checkpoint.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.device = DEVICE
        self.model = model.to(self.device)

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            min_lr=1e-6,
            factor=0.5,
            # verbose=False,  # removed in torch 2.4.0
            patience=2,
            mode='min'
        )

        self.loss = torch.nn.NLLLoss()

        self.best_model = {'model': None, 'val_acc': 0, 'val_loss': float('inf')}

    def training_loop(self):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        for epoch_indx in range(self.num_epochs):
            tl, ta = self.train_one_epoch()
            vl, va, cm = self.validation()

            # print current epoch results:
            print("\n" + "-" * 25 + f" Epoch {epoch_indx + 1} " + "-" * 25)
            print(f"Val Loss = {np.mean(vl):.5f}")
            print(f"Val Accuracy = {100 * np.mean(va):.2f}%")

            # update according to the new results:
            train_loss += [np.mean(tl)]
            train_acc += [np.mean(ta)]
            val_loss += [np.mean(vl)]
            val_acc += [np.mean(va)]

            # scheduler
            # if the val_loss doesn't improve for X epochs, then the learning-rate will be reduced by factor 0.5
            self.scheduler.step(np.mean(vl))
            print(f"Last Learning Rate: {self.scheduler.optimizer.param_groups[0]['lr']}")

            # update plots
            self.apply_plot(
                [train_loss, val_loss],
                ['train_loss', 'val_loss'],
                'Training vs Validation Loss',
                'loss_curves.jpg'
            )
            self.apply_plot(
                [train_acc, val_acc],
                ['train_acc', 'val_acc'],
                'Training vs Validation Accuracy',
                'accuracy_curves.jpg'
            )
            self.plot_confusion_matrix(cm)

            # save the best model so far
            if np.mean(va) >= self.best_model['val_acc']:
                self.best_model['model'] = deepcopy(self.model)
                self.best_model['val_acc'] = np.mean(va)
                self.best_model['val_loss'] = np.mean(vl)
                self.save_best_model()

    def train_one_epoch(self):
        self.model.train()
        training_loss = []
        train_acc = []
        for batch in self.training_dataloader:
            waveforms, labels = batch
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            probs = self.model.forward(waveforms)
            log_probs = torch.log(probs + 1e-5)
            loss = self.loss(log_probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            training_loss.append(loss.item())
            train_acc.append((torch.argmax(probs, dim=-1) == labels).type(torch.float32).mean().item())

        return training_loss, train_acc

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        val_loss = []
        val_acc = []
        confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=self.device)
        for batch in self.val_dataloader:
            waveforms, labels = batch
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            probs = self.model.forward(waveforms)
            log_probs = torch.log(probs + 1e-5)
            loss = self.loss(log_probs, labels)

            val_loss.append(loss.item())
            val_acc.append((torch.argmax(probs, dim=-1) == labels).type(torch.float32).mean().item())

            # upadte confusion matrix
            gt_labels = torch.flatten(labels)
            predicted_labels = torch.flatten(torch.argmax(probs, dim=-1))
            linear_index = gt_labels * NUM_CLASSES + predicted_labels
            counts = torch.bincount(linear_index, minlength=NUM_CLASSES ** 2)
            confusion_matrix += counts.view(NUM_CLASSES, NUM_CLASSES)

        # normalized the confusion matrix:
        confusion_matrix /= confusion_matrix.sum(dim=-1, keepdims=True)

        return val_loss, val_acc, confusion_matrix

    @torch.no_grad()
    def testing(self):
        best_model = self.best_model['model'].eval()
        test_loss = []
        test_acc = []
        confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=self.device)
        for batch in self.test_dataloader:
            waveforms, labels = batch
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            probs = best_model.forward(waveforms)
            log_probs = torch.log(probs + 1e-5)
            loss = self.loss(log_probs, labels)

            test_loss.append(loss.item())
            test_acc.append((torch.argmax(probs, dim=-1) == labels).type(torch.float32).mean().item())

            # upadte confusion matrix
            gt_labels = torch.flatten(labels)
            predicted_labels = torch.flatten(torch.argmax(probs, dim=-1))
            linear_index = gt_labels * NUM_CLASSES + predicted_labels
            counts = torch.bincount(linear_index, minlength=NUM_CLASSES ** 2)
            confusion_matrix += counts.view(NUM_CLASSES, NUM_CLASSES)

        # normalized the confusion matrix:
        confusion_matrix /= confusion_matrix.sum(dim=-1, keepdims=True)

        return test_loss, test_acc, confusion_matrix

    def apply_plot(self, curves, labels, title=None, output_filename=None):
        fig, ax = plt.subplots(1)
        for curve, label in zip(curves, labels):
            ax.plot(curve, label=label)
        ax.legend()
        ax.grid(True)
        if title:
            ax.set_title(title)
        if output_filename:
            assert str(output_filename).endswith('.jpg')
            plt.savefig(self.checkpoint / output_filename, dpi=200)
        plt.clf()
        plt.close()

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(16, 12))
        plt.imshow(cm.cpu().numpy())
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, [I2L[i] for i in range(NUM_CLASSES)], rotation=45)
        plt.yticks(tick_marks, [I2L[i] for i in range(NUM_CLASSES)])

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        plt.savefig(self.checkpoint / 'confusion_matrix.jpg', dpi=400)
        plt.clf()
        plt.close()

    def save_best_model(self):
        states = {
            'state_dict': self.best_model['model'].state_dict(),
            'val_acc': self.best_model['val_acc'],
            'val_loss': self.best_model['val_loss']
        }
        torch.save(states, self.checkpoint / 'best_model.pt')


if __name__ == '__main__':
    model = DeepGSC(
        num_classes=len(labels),
        sample_rate=16_000,
        window_len=0.032,
        hop_size=0.01,
        nfft=512,
        num_mel=32
    )
    model.print_model_weights()

    trainer = Trainer(
        model=model,
        checkpoint=Path('./exp_1'),
        batch_size=64,
        learning_rate=1e-4,
        num_epochs=50
    )
    trainer.training_loop()

    model.print_model_weights()

    # load the best model for testing:
    best_model = deepcopy(model)
    best_model.load_state_dict(torch.load('./exp_1/best_model.pt', map_location=DEVICE)['state_dict'])
    trainer.best_model = {'model': best_model}

    test_loss, test_acc, test_cm = trainer.testing()
    print("\n" + "-" * 50)
    print(f"Test Loss = {np.mean(test_loss):.5f}")
    print(f"Test Accuracy = {100 * np.mean(test_acc):.2f}%")
    print("-" * 50)

    trainer.plot_confusion_matrix(test_cm)

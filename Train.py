import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, f1_score
from torch.utils.data import random_split
from sample.script import SSDataset_690
from Utils.EarlyStopping import EarlyStopping
from model.Hample import Hample


class Trainer:

    def __init__(self, model, model_name, TF,
                 batch_size, epochs, cell_num):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.TF = TF
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()

        self.batch_size = batch_size
        self.epochs = epochs
        self.cell_num = cell_num

        self.sequence_order = 3  # default(3) empiric

    def learn(self, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedModels"
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                sequence, shape, epigenome, labels = data

                # cell_num, bs, 1
                binding_predictions = self.model(sequence.to(self.device, dtype=torch.float),
                                                 shape.to(self.device, dtype=torch.float),
                                                 epigenome.to(self.device, dtype=torch.float))
                # cell_num, bs
                labels = labels.permute(1, 0)
                final_loss = 0
                for prediction_in_bs, label_in_bs in zip(binding_predictions, labels):
                    final_loss = final_loss + self.loss_function(prediction_in_bs,
                                                                 label_in_bs.float().to(self.device))

                final_loss = final_loss / self.cell_num
                ProgressBar.set_postfix(loss=final_loss.item())

                final_loss.backward()
                self.optimizer.step()

            final_valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_sequence, valid_shape, valid_epigenome, valid_labels in ValidateLoader:
                    # cell_num, bs, 1
                    valid_binding_predictions = self.model(valid_sequence.to(self.device, dtype=torch.float),
                                                           valid_shape.to(self.device, dtype=torch.float),
                                                           valid_epigenome.to(self.device, dtype=torch.float))
                    # cell_num, bs
                    valid_labels = valid_labels.float().to(self.device)
                    valid_labels = valid_labels.permute(1, 0)

                    valid_loss_in_bs = 0
                    for valid_prediction_in_bs, valid_label_in_bs in zip(valid_binding_predictions, valid_labels):
                        valid_loss_in_bs = valid_loss_in_bs + self.loss_function(valid_prediction_in_bs,
                                                                                 valid_label_in_bs).item()

                    final_valid_loss.append(valid_loss_in_bs / self.cell_num)

                valid_loss_avg = torch.mean(torch.Tensor(final_valid_loss))
                self.scheduler.step(valid_loss_avg)

            early_stopping(valid_loss_avg, self.model,
                           path + '\\' + self.TF + '.pth')

        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):

        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedModels"

        self.model.load_state_dict(torch.load(path + '\\' + self.TF + '.pth', map_location='cpu'))
        self.model.to("cpu")

        predicted_values = []
        ground_labels = []
        self.model.eval()

        for sequence, shape, epigenome, labels in TestLoader:
            # bs=1 (default)
            # cell_num, bs, 1
            binding_predictions = self.model(sequence.float(), shape.float(), epigenome.float())
            # cell_num, bs
            labels = labels.permute(1, 0)
            for prediction, label in zip(binding_predictions, labels):
                """ To scalar"""
                predicted_values.append(prediction.squeeze(dim=0).squeeze(dim=0).detach().numpy())
                ground_labels.append(label.squeeze(dim=0).detach().numpy())

        print('\n---Finish Inference---\n')

        return predicted_values, ground_labels

    def measure(self, predicted_values, ground_labels):
        accuracy = accuracy_score(y_pred=np.array(predicted_values).round(), y_true=ground_labels)
        roc_auc = roc_auc_score(y_score=predicted_values, y_true=ground_labels)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_values, y_true=ground_labels)
        pr_auc = auc(recall, precision)

        f_score = f1_score(y_pred=np.array(predicted_values).round(), y_true=ground_labels)

        print('\n---Finish Measure---\n')

        return accuracy, roc_auc, pr_auc, f_score

    def save_evaluation_indicators(self, indicators):
        path = os.path.abspath(os.curdir) + "\\" + self.model_name + "SavedIndicators"

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = path + "\\" + self.model_name + "Indicators.xlsx"
        file = open(file_name, "a")

        file.write(str(indicators[0]) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + " " +
                   str(np.round(indicators[4], 4)) + "\n")

        file.close()

    def run(self, samples_file_name, ratio=0.8):
        """
        Train_Validate_Set = SSDataset_690(samples_file_name, self.sequence_order, False)
        """
        Train_Validate_Set = SSDataset_690(samples_file_name, self.sequence_order, True)

        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, self.sequence_order, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        self.learn(TrainLoader, ValidateLoader)

        predicted_values, ground_labels = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc, f_score = self.measure(predicted_values, ground_labels)

        # 写入评价指标
        indicators = [self.TF, accuracy, roc_auc, pr_auc, f_score]
        self.save_evaluation_indicators(indicators)

        print('\n---Finish Run---\n')


def main():
    TFs = ['ATF2', 'ATF3', 'BHLHE40', 'CEBPB', 'CTCF', 'EGR1', 'ELF1', 'EZH2', 'FOS',
           'GABPA', 'GATA2', 'GTF2F1', 'HDAC2', 'JUN', 'JUND', 'MAFK', 'MAX', 'MAZ',
           'MXI1', 'MYC', 'NRF1', 'RAD21', 'REST', 'RFX5', 'SIN3A', 'SMC3', 'SP1', 'SRF',
           'SUZ12', 'TAF1', 'TCF12', 'TEAD4', 'TCF7L2', 'USF1', 'USF2', 'YY1']

    for TF in TFs:
        Train = Trainer(model=Hample(),
                        TF=TF, model_name='Hample', batch_size=64, epochs=15, cell_num=5)
        Train.run(samples_file_name=TF)


main()


"""
Train = Trainer(model=Hample(),
                TF='ATF2', model_name='Hample', batch_size=1, epochs=15, cell_num=5)
Train.run(samples_file_name='USF2')
"""
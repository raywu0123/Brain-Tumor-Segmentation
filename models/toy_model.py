import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .base import Segmentation2DModelBase
from utils import MetricClass

class ToyModel(Segmentation2DModelBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            num_units: [int] = (64, 128, 256, 512),
            kernel_size: int = 5,    
        ):
        self.num_units = num_units
        self.data_channels = channels
        self.data_depth = depth
        self.data_height = height
        self.data_width = width
        self.metadata_dim = metadata_dim

        self.model = ToyModelNet(
            image_chns=self.data_channels,
            image_height=self.data_height,
            image_width=self.data_width,
            num_units=num_units,
            kernel_size=kernel_size,
            pooling_num=3,
        )
        self.opt = optim.Adam(params=self.model.parameters())
        if torch.cuda.is_available():
            self.model.cuda()

    def fit_generator(self, training_data_generator, validation_data_generator, **kwargs):
        print(kwargs)
        batch_size = kwargs['batch_size']
        epoch_num = kwargs['epoch_num']
        
        verbose_epoch_num = kwargs['verbose_epoch_num']
        for i_epoch in range(epoch_num):
            losses, dice_scores = self.train_on_batch(training_data_generator, batch_size)
            
            if i_epoch % verbose_epoch_num == 0:
                print(f'epoch: {i_epoch}, bce_loss: {np.mean(losses)}, dice_score: {np.mean(dice_scores)}')
                self._validate(validation_data_generator, batch_size, verbose_epoch_num // 10)

    def train_on_batch(self, training_data_generator, batch_size):
        image, label = self._get_data_with_generator(
            training_data_generator,
            1,
        )
        losses = []
        dice_scores = []
        
        for batch_idx in range(self.data_depth // batch_size):
            self.model.zero_grad()
            batch_image = image[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_label = label[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_image = torch.Tensor(batch_image)
            batch_label = torch.Tensor(batch_label)
            if torch.cuda.is_available():
                batch_image = batch_image.cuda()
                batch_label = batch_label.cuda()
            batch_size = batch_image.shape[0]

            # pred = self.model(batch_label)
            # pred = batch_label
            pred = self.model(batch_image)
            bce_loss = nn.BCELoss()(pred, batch_label)

            smooth = 1
            m1 = pred.view(batch_size, -1)
            m2 = batch_label.view(batch_size, -1)
            intersection = (m1 * m2)
            m1 = torch.sum(m1, dim=1)
            m2 = torch.sum(m2, dim=1)
            intersection = torch.sum(intersection, dim=1)
            
            dice_score = 2. * (intersection + smooth) / (m1 + m2 + smooth)
            dice_score = dice_score.sum() / batch_size
            
            total_loss = -torch.log(dice_score)
            # total_loss.backward()
            # self.opt.step()

            losses.append(bce_loss.item())
            dice_scores.append(dice_score.item())

        return losses, dice_scores

    def _validate(self, validation_data_generator, batch_size, verbose_epoch_num):
        label_buff = []
        pred_buff = []
        
        for batch_num in range(verbose_epoch_num):
            image, label = self._get_data_with_generator(
                validation_data_generator,
                1,
            )
            for batch_idx in range(self.data_depth // batch_size):
                batch_image = image[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_label = label[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_image = torch.Tensor(batch_image)
                if torch.cuda.is_available():
                    batch_image = batch_image.cuda()
                batch_pred = self.model(batch_image)
                batch_pred = batch_pred.cpu().data.numpy()

                label_buff.extend(batch_label)
                pred_buff.extend(batch_pred)
        
        label = np.asarray(label_buff)
        pred = np.asarray(pred_buff)
        pred = (pred > 0.5).astype(int)
        metric = MetricClass(
            pred,
            label,
        ).all_metrics()

    def _get_data_with_generator(self, generator, batch_size):
        batch_data = generator(batch_size=batch_size)
        batch_image, batch_label = batch_data['img'], batch_data['label']
        batch_image = np.transpose(batch_image, (4, 0, 1, 2, 3))
        batch_label = np.transpose(batch_label,(4, 0, 1, 2, 3))
        batch_image = np.squeeze(batch_image, axis=1)
        batch_label = np.squeeze(batch_label, axis=1)
        # np.random.shuffle(batch_image)
        # np.random.shuffle(batch_label)

        batch_image = batch_image - np.mean(batch_image, axis=0)
        return batch_image, batch_label

    def predict(self, x_test, **kwargs):
        x_test = torch.Tensor(x_test)
        if torch.cuda.is_available():
            x_test = x_test.cuda()

        pred = self.model(x_test)
        return pred.cpu().data.numpy()


class ToyModelNet(nn.Module):
    def __init__(self, image_chns, image_height, image_width, num_units, kernel_size, pooling_num):
        super(ToyModelNet, self).__init__()
        self.image_chns = image_chns
        self.image_height = image_height
        self.image_width = image_width

        encoder_num_units = (image_chns,) + num_units
        self.encoder_convs = nn.ModuleList()
        self.encoder_batchnorms = nn.ModuleList()
        for idx in range(len(encoder_num_units) - 1):
            if idx < pooling_num:
                stride = 2
            else:
                stride = 1
            in_chns = encoder_num_units[idx]
            out_chns = encoder_num_units[idx + 1]
            conv = nn.Conv2d(
                in_chns,
                out_chns,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            batchnorm = nn.BatchNorm2d(in_chns)
            self.encoder_convs.append(conv)
            self.encoder_batchnorms.append(batchnorm)

        decoder_num_units = num_units[::-1] + (image_chns,)
        self.decoder_deconvs = nn.ModuleList()
        self.decoder_batchnorms = nn.ModuleList()
        img_size = image_height // 4
        for idx in range((len(decoder_num_units))- 1):
            if idx < pooling_num:
                stride = 2
                img_size = img_size * 2
                output_padding = 1
            else:
                stride = 1
                output_padding = 0
            deconv = nn.ConvTranspose2d(
                decoder_num_units[idx],
                decoder_num_units[idx + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=output_padding,
            )
            batchnorm = nn.BatchNorm2d(decoder_num_units[idx])
            self.decoder_deconvs.append(deconv)
            self.decoder_batchnorms.append(batchnorm)

    def forward(self, inp):
        x = inp
        for conv, batchnorm in zip(self.encoder_convs, self.encoder_batchnorms):
            # x = batchnorm(x)
            x = F.relu(x)
            x = conv(x)

        for deconv, batchnorm in zip(self.decoder_deconvs, self.decoder_batchnorms):
            # x = batchnorm(x)
            x = F.relu(x)
            x = deconv(x)
        
        x = torch.sigmoid(x)
        return x

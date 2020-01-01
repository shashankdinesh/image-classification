import torch.nn as nn


class RiceClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super(RiceClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu10 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu12 = nn.ReLU()

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu13 = nn.ReLU()

        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.lf1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True)
        self.relu14 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.lf2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu15 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.lf3 = nn.Linear(in_features=4096, out_features=2, bias=True)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batchnorm1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.relu2(output)

        output = self.maxpool1(output)

        output = self.conv3(output)
        output = self.batchnorm3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.batchnorm4(output)
        output = self.relu4(output)

        output = self.maxpool2(output)

        output = self.conv5(output)
        output = self.batchnorm5(output)
        output = self.relu5(output)

        output = self.conv6(output)
        output = self.batchnorm6(output)
        output = self.relu6(output)

        output = self.conv7(output)
        output = self.batchnorm7(output)
        output = self.relu7(output)

        output = self.maxpool3(output)

        output = self.conv8(output)
        output = self.batchnorm8(output)
        output = self.relu8(output)

        output = self.conv9(output)
        output = self.batchnorm9(output)
        output = self.relu9(output)

        output = self.conv10(output)
        output = self.batchnorm10(output)
        output = self.relu10(output)

        output = self.maxpool4(output)

        output = self.conv11(output)
        output = self.batchnorm11(output)
        output = self.relu11(output)

        output = self.conv12(output)
        output = self.batchnorm12(output)
        output = self.relu12(output)

        output = self.conv13(output)
        output = self.batchnorm13(output)
        output = self.relu13(output)

        output = self.maxpool5(output)
        output = output.view(-1, 7 * 7 * 512)

        output = self.lf1(output)
        output = self.relu14(output)
        output = self.dropout1(output)

        output = self.lf2(output)
        output = self.relu15(output)
        output = self.dropout2(output)

        output = self.lf3(output)

        return output
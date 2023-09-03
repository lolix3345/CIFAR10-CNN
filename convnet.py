import torch.nn as nn

class CNN_modulable_width(nn.Module):
    def __init__(self, width=16):
        super(CNN_modulable_width, self).__init__()
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=3,
                          out_channels=self.width,
                          kernel_size=7,
                          stride=1,
                          padding=3)
        self.conv2 = nn.Conv2d(in_channels=self.width,
                          out_channels=self.width*2,
                          kernel_size=3,
                          stride=2,
                          padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.width*2,
                          out_channels=self.width*2,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.width*2,
                          out_channels=self.width*4,
                          kernel_size=3,
                          stride=2,
                          padding=1)
        self.conv5 = nn.Conv2d(in_channels=self.width*4,
                          out_channels=self.width*4,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        self.conv6 = nn.Conv2d(in_channels=self.width*4,
                          out_channels=self.width*8,
                          kernel_size=3,
                          stride=2,
                          padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4))
        self.final_classifier = nn.Sequential(
            nn.Linear(self.width*8,self.width*32),
            nn.ReLU(),
            nn.Linear(self.width*32,10)
        )

    def forward(self, input):
        out1 = nn.functional.relu(self.conv1(input))
        out2 = nn.functional.relu(self.conv2(out1))
        out3 = nn.functional.relu(self.conv3(out2))
        out4 = nn.functional.relu(self.conv4(out3))
        out5 = nn.functional.relu(self.conv5(out4))
        out6 = nn.functional.relu(self.conv6(out5))
        out_conv = self.avgpool(out6).squeeze()
        #print(out1.size(),out2.size(),out3.size(),out4.size(),out5.size(),out6.size(),out_conv.size())
        out_classifier = self.final_classifier(out_conv)
        return out_classifier

class Depthwise_separable_CNN_modulable_width(nn.Module):
    def __init__(self, width=16):
        super(Depthwise_separable_CNN_modulable_width, self).__init__()
        self.width = width
        self.conv1 = nn.Conv2d(in_channels=3,
                          out_channels=self.width,
                          kernel_size=7,
                          stride=1,
                          padding=3)

        self.depthwise_separable_conv2 = Depthwise_Separable_Block(in_features=self.width,
                                                               out_features=self.width * 2,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1)

        self.depthwise_separable_conv3 = Depthwise_Separable_Block(in_features=self.width * 2,
                                                               out_features=self.width * 2,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1)

        self.depthwise_separable_conv4 = Depthwise_Separable_Block(in_features=self.width * 2,
                                                                   out_features=self.width * 4,
                                                                   kernel_size=3,
                                                                   stride=2,
                                                                   padding=1)

        self.depthwise_separable_conv5 = Depthwise_Separable_Block(in_features=self.width * 4,
                                                                   out_features=self.width * 4,
                                                                   kernel_size=3,
                                                                   stride=1,
                                                                   padding=1)

        self.depthwise_separable_conv6 = Depthwise_Separable_Block(in_features=self.width * 4,
                                                                   out_features=self.width * 8,
                                                                   kernel_size=3,
                                                                   stride=2,
                                                                   padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=(4,4))
        self.final_classifier = nn.Sequential(
            nn.Linear(self.width*8,self.width*32),
            nn.ReLU(),
            nn.Linear(self.width*32,10)
        )

    def forward(self, input):
        out1 = nn.functional.relu(self.conv1(input))
        out2 = nn.functional.relu(self.depthwise_separable_conv2(out1))
        out3 = nn.functional.relu(self.depthwise_separable_conv3(out2))
        out4 = nn.functional.relu(self.depthwise_separable_conv4(out3))
        out5 = nn.functional.relu(self.depthwise_separable_conv5(out4))
        out6 = nn.functional.relu(self.depthwise_separable_conv6(out5))
        out_conv = self.avgpool(out6).squeeze()
        #print(out1.size(),out2.size(),out3.size(),out4.size(),out5.size(),out6.size(),out_conv.size())
        out_classifier = self.final_classifier(out_conv)
        return out_classifier

class Depthwise_Separable_Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super(Depthwise_Separable_Block, self).__init__()
        self.separable_conv = nn.Conv2d(in_channels=in_features,
                                        out_channels=in_features,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        groups=in_features)

        self.pointwise_conv = nn.Conv2d(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
    def forward(self, input):
        return self.pointwise_conv(self.separable_conv(input))

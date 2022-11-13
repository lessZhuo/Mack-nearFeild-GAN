import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import functools


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                kernel_size=3, stride=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True),
                      nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2) * 4,
                                kernel_size=1, stride=1),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True),
                      ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        attention_mask = F.sigmoid(output[:, :1])
        content_mask = output[:, 1:]
        attention_mask = attention_mask.repeat(1, 3, 1, 1)
        result = content_mask * attention_mask + input * (1 - attention_mask)

        return result, attention_mask, content_mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout)

    def build_conv_block(self, dim, norm_layer, use_dropout):
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3),
                      norm_layer(dim),
                      nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_tower = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1, 1),
        )

    def forward(self, img):
        output = self.conv_tower(img)
        return output


class ResnetGenerator_Attention(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_Attention, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks1.weight_init(0, 0.02)
        # self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks2.weight_init(0, 0.02)
        # self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks3.weight_init(0, 0.02)
        # self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks4.weight_init(0, 0.02)
        # self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks5.weight_init(0, 0.02)
        # self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks6.weight_init(0, 0.02)
        # self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks7.weight_init(0, 0.02)
        # self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks8.weight_init(0, 0.02)
        # self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks9.weight_init(0, 0.02)

        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        # -----------这里的27是根据3通道扩展9倍 最好设置为chnnal*9---------------------------
        self.deconv3_content = nn.Conv2d(ngf, 27, 7, 1, 0)

        self.deconv3_content = nn.Conv2d(ngf, self.output_nc * 10, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        # ---------------------------10修改为9，减少通道参数-------------------------
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)

        self.tanh = torch.nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        # x = self.resnet_blocks1(x)
        # x = self.resnet_blocks2(x)
        # x = self.resnet_blocks3(x)
        # x = self.resnet_blocks4(x)
        # x = self.resnet_blocks5(x)
        # x = self.resnet_blocks6(x)
        # x = self.resnet_blocks7(x)
        # x = self.resnet_blocks8(x)
        # x = self.resnet_blocks9(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        image = self.tanh(content)
        # ----------------------------------------------------这里要修改通道 原本图像是3通道，要改为2通道
        # image1 = image[:, 0:3, :, :]
        # image2 = image[:, 3:6, :, :]
        # image3 = image[:, 6:9, :, :]
        # image4 = image[:, 9:12, :, :]
        # image5 = image[:, 12:15, :, :]
        # image6 = image[:, 15:18, :, :]
        # image7 = image[:, 18:21, :, :]
        # image8 = image[:, 21:24, :, :]
        # image9 = image[:, 24:27, :, :]

        image1 = image[:, self.output_nc * 0: self.output_nc * 1, :, :]
        image2 = image[:, self.output_nc * 1: self.output_nc * 2, :, :]
        image3 = image[:, self.output_nc * 2: self.output_nc * 3, :, :]
        image4 = image[:, self.output_nc * 3: self.output_nc * 4, :, :]
        image5 = image[:, self.output_nc * 4: self.output_nc * 5, :, :]
        image6 = image[:, self.output_nc * 5: self.output_nc * 6, :, :]
        image7 = image[:, self.output_nc * 6: self.output_nc * 7, :, :]
        image8 = image[:, self.output_nc * 7: self.output_nc * 8, :, :]
        image9 = image[:, self.output_nc * 8: self.output_nc * 9, :, :]
        image10 = image[:, self.output_nc * 9: self.output_nc * 10, :, :]

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, self.output_nc, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, self.output_nc, 1, 1)
        attention3 = attention3_.repeat(1, self.output_nc, 1, 1)
        attention4 = attention4_.repeat(1, self.output_nc, 1, 1)
        attention5 = attention5_.repeat(1, self.output_nc, 1, 1)
        attention6 = attention6_.repeat(1, self.output_nc, 1, 1)
        attention7 = attention7_.repeat(1, self.output_nc, 1, 1)
        attention8 = attention8_.repeat(1, self.output_nc, 1, 1)
        attention9 = attention9_.repeat(1, self.output_nc, 1, 1)
        attention10 = attention10_.repeat(1, self.output_nc, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        output10 = image10 * attention10
        # output10 = input * attention10

        o = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        return o, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1, attention2, attention3, attention4, attention5, attention6, attention7, attention8, attention9, attention10, image1, image2, image3, image4, image5, image6, image7, image8, image9


# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='batch'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

import torch.nn as nn
import torch.nn.functional as F
import torch


##############################
#      SELF-ATTENTION
##############################

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, output_shape, num_residual_blocks, fw=True):
        super(GeneratorResNet, self).__init__()
        self.fw = fw
        input_channels = input_shape[0]
        output_channels = output_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channels, 16, 2),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_features, 2),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        out_features *= 2
        layer_down_1 = [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        self.attn1 = Self_Attn(out_features, 'relu')

        in_features = out_features
        out_features *= 2
        layer_down_2 = [
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        self.attn2 = Self_Attn(out_features, 'relu')
        in_features = out_features

        # Residual blocks
        model_res = []
        for _ in range(num_residual_blocks):
            model_res += [ResidualBlock(out_features)]

        # Upsampling
        out_features //= 2
        layer_up_1 = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        self.attn3 = Self_Attn(out_features, 'relu')
        in_features = out_features
        out_features //= 2
        layer_up_2 = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        self.attn4 = Self_Attn(out_features, 'relu')
        in_features = out_features

        # Output layer
        model_out = [nn.ReflectionPad2d(1),
                     nn.Conv2d(out_features, 16, 2),
                     nn.Tanh(),
                     nn.Conv2d(16, output_channels, 2)]

        if fw:
            model_out += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.layer_down_1 = nn.Sequential(*layer_down_1)

        self.layer_down_2 = nn.Sequential(*layer_down_2)

        self.model_res = nn.Sequential(*model_res)
        self.layer_up_1 = nn.Sequential(*layer_up_1)

        self.layer_up_2 = nn.Sequential(*layer_up_2)

        self.model_out = nn.Sequential(*model_out)

    def forward(self, x):

        x = self.model(x)
        x = self.layer_down_1(x)
        x, a1 = self.attn1(x)
        x = self.layer_down_2(x)
        x, a2 = self.attn2(x)
        x = self.model_res(x)
        x = self.layer_up_1(x)
        x, a3 = self.attn3(x)
        x = self.layer_up_2(x)
        x, a4 = self.attn4(x)
        x = self.model_out(x)

        if self.fw:
            return x
        else:
            x = F.log_softmax(x, dim=1)
            return x


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_in = nn.Sequential(
            *discriminator_block(channels, 16, normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 128),
        )
        self.attn = Self_Attn(128, 'relu')
        self.model_out = nn.Sequential(
            *discriminator_block(128, 256),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 1, 3)  # , padding=1)
        )

    def forward(self, x):
        x = self.model_in(x)
        x, a = self.attn(x)
        x = self.model_out(x)
        return x

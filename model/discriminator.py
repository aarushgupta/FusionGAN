import torch
import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        
        use_bias = norm_layer

        kw = 3
        padw = 1
        sequence = []
        self.initial_conv = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.initial_relu = nn.LeakyReLU(0.2, True)
        
    
        ndf = 128
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        
        self.maxpool = nn.MaxPool2d(8)
    def forward(self, x, y):
        x = self.initial_relu(self.initial_conv(x))
        y = self.initial_relu(self.initial_conv(y))
        
        concat_fmap = torch.zeros([x.shape[0], 2 * x.shape[1], x.shape[2], x.shape[3]], dtype=x.dtype)
#         print(x.shape, y.shape, concat_result.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                concat_fmap[i][j] = x[i][j]
                concat_fmap[i][j + 64] = y[i][j]
        if torch.cuda.is_available():
        	concat_fmap = concat_fmap.cuda()
        op = self.model(concat_fmap)
        op_neg = - op
        op_neg = self.maxpool(op_neg)
        op_pooled = -op_neg
        
        return op, op_pooled
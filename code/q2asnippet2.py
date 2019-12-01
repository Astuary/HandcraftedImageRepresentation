self.decoder = nn.Sequential(
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.ConvTranspose2d(64, 32, 3, stride=1),  # b, 16, 5, 5
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  # b, 8, 15, 15
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Upsample(mode='bilinear', scale_factor=2),
        nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),  # b, 1, 28, 28
        nn.BatchNorm2d(1),
        nn.Tanh(),
        nn.Upsample(mode='bilinear', scale_factor=2)
    )

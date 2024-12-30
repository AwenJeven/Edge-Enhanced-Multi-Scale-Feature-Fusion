class Eattention(nn.Layer):
    def __init__(self, hist_channel):
        super(Eattention, self).__init__()
        self.fcq_hist1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fcq_hist2 = nn.Conv2D(64, 8, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.fcq_hist3 = nn.Conv2D(8, 64, kernel_size=1, stride=1, padding=0)

        prewitt_horizontal = paddle.to_tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype='float32').unsqueeze(0).unsqueeze(0)
        prewitt_vertical = paddle.to_tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float32').unsqueeze(0).unsqueeze(0)
        sobel_horizontal = paddle.to_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32').unsqueeze(0).unsqueeze(0)
        sobel_vertical = paddle.to_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32').unsqueeze(0).unsqueeze(0)
        laplacian = paddle.to_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32').unsqueeze(0).unsqueeze(0)

        self.conv1_weight = prewitt_horizontal.tile([64, 64, 1, 1])
        self.conv2_weight = prewitt_vertical.tile([64, 64, 1, 1])
        self.conv3_weight = sobel_horizontal.tile([64, 64, 1, 1])
        self.conv4_weight = sobel_vertical.tile([64, 64, 1, 1])
        self.conv5_weight = laplacian.tile([64, 64, 1, 1])
        self.conv1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False)
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False)
        self.conv5 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False)
        self.conv1.weight.set_value(self.conv1_weight)
        self.conv2.weight.set_value(self.conv2_weight)
        self.conv3.weight.set_value(self.conv3_weight)
        self.conv4.weight.set_value(self.conv4_weight)
        self.conv5.weight.set_value(self.conv5_weight)

        self.conv6 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias_attr=False, stride=1)
        self.conv7 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, padding=2, bias_attr=False, stride=1)
        self.conv8 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, padding=3, bias_attr=False, stride=1)

    def forward(self, x):
        b, c, h, w = x.shape
        #print('1')
        # data_hist = paddle.full([b, 3, 1, 8], 0.125)  
        # q_hist = self.fcq_hist3(self.relu2(self.fcq_hist2(self.relu1(self.fcq_hist1(data_hist)))))
        paddle.save(x, 'input_image.pdparams')
        input_image_np = x.numpy()
        np.save('input_image.npy', input_image_np[0, 0, :, :])  
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        x7 = self.conv7(x)
        x8 = self.conv8(x)

        y = (x1 * 0.2) + (x2 * 0.2) + (x3 * 0.2) + (x4 * 0.2) + \
            (x5 * 0.2)

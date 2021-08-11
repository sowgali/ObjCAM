'''
    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [160, 160, 1]
        assert green.get_shape().as_list()[1:] == [160, 160, 1]
        assert blue.get_shape().as_list()[1:] == [160, 160, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [160, 160, 3]

        self.conv1_1 = self.conv_layer(bgr, 'block1_conv1')
        self.conv1_2 = self.conv_layer(self.conv1_1, "block1_conv2")
        self.pool1 = self.max_pool(self.conv1_2, 'block1_pool')

        self.conv2_1 = self.conv_layer(self.pool1, "block2_conv1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "block2_conv2")
        self.pool2 = self.max_pool(self.conv2_2, 'block2_pool')

        self.conv3_1 = self.conv_layer(self.pool2, "block3_conv1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "block3_conv2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "block3_conv3")
        self.pool3 = self.max_pool(self.conv3_3, 'block3_pool')

        self.conv4_1 = self.conv_layer(self.pool3, "block4_conv1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "block4_conv2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "block4_conv3")
        self.pool4 = self.max_pool(self.conv4_3, 'block4_pool')

        self.conv5_1 = self.conv_layer(self.pool4, "block5_conv1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "block5_conv2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "block5_conv3")
        self.pool5 = self.max_pool(self.conv5_3, 'block5_pool')

        self.fc6 = self.fc_layer(self.pool5, "dense")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "dense_1")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "dense_2")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))
'''

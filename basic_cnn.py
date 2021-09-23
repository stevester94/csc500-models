#! /usr/bin/env python3

import torch.nn as nn

class Basic_CNN_Model(nn.Module):
	def __init__(self,
        num_output_classes):
		super(Basic_CNN_Model, self).__init__()

		self.conv = nn.Sequential()
		self.dense = nn.Sequential()

		# Unique naming matters
		
		# This first layer does depthwise convolution; each channel gets (out_channels/groups) number of filters. These are applied, and
		# then simply stacked in the output
		self.conv.add_module('dyuh_1', nn.Conv1d(in_channels=2, out_channels=50, kernel_size=7, stride=1))
		self.conv.add_module('dyuh_2', nn.ReLU(False)) # Optionally do the operation in place
		self.conv.add_module('dyuh_3', nn.Conv1d(in_channels=50, out_channels=50, kernel_size=7, stride=2))
		self.conv.add_module('dyuh_4', nn.ReLU(False))
		self.conv.add_module('dyuh_5', nn.Dropout())
		self.conv.add_module("dyuh_6", nn.Flatten())

		self.dense.add_module('dyuh_7', nn.Linear(50 * 58, 80)) # Input shape, output shape
		self.dense.add_module('dyuh_8', nn.ReLU(False))
		self.dense.add_module('dyuh_9', nn.Dropout())
		self.dense.add_module('dyuh_10', nn.Linear(80, num_output_classes))
		self.dense.add_module('dyuh_11', nn.LogSoftmax(dim=1))

	def forward(self, x):
		conv_result = self.conv(x)
		y_hat = self.dense(conv_result)
		return y_hat

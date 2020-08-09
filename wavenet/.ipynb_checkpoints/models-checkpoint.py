#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conditional_WaveNet(nn.Module):
    def __init__(self,
                layer_size=10,
                stack_size=1,
                residual_channels = 64,
                skip_connections = 128,
                input_chan = 12,
                out_chan = 8,
                bias=False):
        super(WaveNet, self).__init__()
        #Hyper parameters
        self.input_chan = input_chan
        self.out_chan = out_chan

        self.residual_channels = residual_channels
        self.skip_connections = skip_connections

        #Build Model
        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.dilations = []
        self.dilated_queues = []


        self.causal_conv = CausalConv1d(in_channels=input_chan, out_channels=residual_channels)
        self.res_stack = ResidualStack(layer_size=layer_size, stack_size=stack_size, 
                                    res_channels=residual_channels, skip_channels=skip_connections)
        #self.last_net = LastNet(128, 256)
        self.last_nets = nn.ModuleList(LastNet(self.skip_connections, 256) for _ in range(self.out_chan))

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(layer_size)] * stack_size
        num_receptive_fields = np.sum(np.unique(np.array(layers)))
        
        return int(num_receptive_fields)
    
    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields
        self.check_input_size(x, output_size)

        return output_size
    
    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise NameError('The data x is too short! The expected output size is {}'.format(output_size))
    
    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        if len(x.size()):
            x = x.squeeze(1)
        output = x.transpose(1,2).contiguous()
        output_size = self.calc_output_size(output)
        
        output = self.causal_conv(output)
        
        skips = self.res_stack(output, output_size)
        
        output = torch.sum(skips, dim=0)
        #output = self.last_net(output)
        decisions  = self.last_nets[0](output).unsqueeze(1)
        
        for i in range(1, len(self.last_nets)):
            o = self.last_nets[i](output).unsqueeze(1)
            decisions = torch.cat([decisions, o], axis=1)

        return decisions.transpose(1, 2).contiguous().transpose(2,3).contiguous()

#%%
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(CausalConv1d, self).__init__()

        # To match size of input and output, set padding = 1
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            padding=1, stride=1, kernel_size=2, bias=bias)

    def forward(self, x):
        #To make size of output same as size of input, remove last value
        x = self.conv(x)[:,:,:-1] #(minibatch, channels, length) 
        return x

#%%
class DilatedConv1d(nn.Module):
    def __init__(self, channels, dilation=1, bias=False):
        super(DilatedConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels,
                            kernel_size=2, stride=1,
                            dilation=dilation, padding=0,
                            bias=bias)
        
    def forward(self, x):
        output = self.conv(x)
        return output

#%%
class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, cond_channels=12, dilation=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.causal_gate = DilatedConv1d(channels=res_channels, dilation=dilation, bias=bias)
        self.causal_filter = DilatedConv1d(channels=res_channels, dilation=dilation, bias=bias)
        
        self.condition_gate = nn.Conv1d(in_channels=cond_channels, out_channels=res_channels, kernel_size=1, stride=1)
        self.condition_filter = nn.Conv1d(in_channels=cond_channels, out_channels=res_channels, kernel_size=1, stride=1)
        
        self.conv_res = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, padding=0, stride=1, kernel_size=1)
        self.conv_skip = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, padding=0, stride=1, kernel_size=1)

        self.gated_tanh = nn.Tanh()
        self.gated_sig = nn.Sigmoid()
    
    def forward(self, x, c, skip_size):
        """
        :param x: Sound data which is predicted previously
        :param c: Conditional inputs => acceleration data
        :param skip_size: The last output size for loss and prediction
        """
        conv_gate   = self.causal_gate(x) + self.condition_gate(c)
        conv_filter = self.causal_filter(x) + self.condition_filter(c)
        
        out = self.gated_tanh(conv_filter) * self.gated_sig(conv_gate)
        
        skip = self.conv_skip(out)
        skip = skip[:,:,-skip_size:]
        
        # forward for residual conv
        input_cut = x[:,:,-output.size(2):]
        res = self.conv_res(out) + input_cut

        return res, skip

#%%
class ResidualStack(nn.Module):
    def __init__(self, layer_size=10, stack_size=5, res_channels=32, skip_channels=1):
        """
        Stack ResidualBlock by layer and stack size
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.skip_channels = skip_channels

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)
    
    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)
        
        if torch.cuda.is_available():
            block.cuda()
        
        return block

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        """
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)
        
        return res_blocks
            
    
    def build_dilations(self):
        dilations = []
        # 5 = stack[layer 1, layer 2, layer 3, layer 4, layer 5]
        for s in range(self.stack_size):
            # 10 = layer[dilation=1, dilation=2,4,8,16,32,64,128,256,512]
            for l in range(self.layer_size):
                dilations.append(2 ** l)
        
        return dilations
    
    def forward(self, x, condition, skip_size):
        """
        :param x:
        :param condition: Condition data => Acceleration data
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            #output is the next input
            output, skip = res_block(output, condition, skip_size)
            skip_connections.append(skip)
        
        return torch.stack(skip_connections)

#%%
class LastNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Last network of the wavenet
        :param channels: number of channels for input and output
        :return:
        """
        super(LastNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        self.conv2 = nn.Conv1d(512, out_channels, 1)
        self.relu = nn.ReLU()
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        #output = self.softmax(output)
        return output

# %%

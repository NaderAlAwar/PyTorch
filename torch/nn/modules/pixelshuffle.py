from .module import Module
from torch.nn.functions.pixelshuffle import PixelShuffleFunction


class PixelShuffle(Module):
    """Rearranges elements in a tensor of shape [*, C*r^2, H, W] to a
    tensor of shape [C, H*r, W*r]. This is useful for implementing
    efficient sub-pixel convolution with a stride of 1/r.
    "Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network" - Shi et. al (2016) for more details
    Args:
        upscale_factor: factor to increase spatial resolution by
    Input Shape: [*, channels*upscale_factor^2, height, width]
    Output Shape:[*, channels, height*upscale_factor, width*upscale_factor]
    Examples:
        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return PixelShuffleFunction(self.upscale_factor)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (upscale_factor=' + str(self.upscale_factor) + ')'

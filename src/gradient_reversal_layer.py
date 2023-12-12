import torch
from torch.autograd import Function


class GradientReversalCore(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha_ = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_

        return grad_input, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalCore.apply(x, alpha)
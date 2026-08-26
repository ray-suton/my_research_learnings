"""Tasks 2 and 3: hand-coded tiny Conv2D forward pass.

Fill this in yourself. Use loops for the manual Conv2D, then compare against
torch.nn.functional.conv2d with the same input, weight, bias, stride, and
padding.
"""

import torch
import torch.nn.functional as F


def manual_conv2d(x, weight, bias, stride=1, padding=0):
    # TODO: Compute output height and width.
    # output (1,1,3,3) we conv with 1 channel, and 3*3 kernel, output_dim = ((4-3) + 2 * 0 ) / 1 + 1 = 2
    output_dim = ((x.shape[2] - weight.shape[2] + 2 * padding)// stride + 1)
    # TODO: Create an output tensor with the right shape.
    output = torch.zeros((1,1,output_dim,output_dim))
    w_shape = weight.shape[2]
    # TODO: Loop over batch, output channel, output row, output column.
    for start_row in range(0, x.shape[2] + padding - w_shape + 1):
        for start_col in range(0, x.shape[2] + padding - w_shape + 1):
            x_dimed = x[:, :, start_row:start_row+w_shape, start_col:start_col+w_shape]
            x_reshaped = torch.reshape(x_dimed,(w_shape, w_shape))
            w_reshaped = torch.reshape(weight,(w_shape, w_shape))
            output[:,:,start_row, start_col] = torch.sum(x_reshaped * w_reshaped)
    
    # TODO: Select the matching input patch.
    # TODO: Multiply patch * kernel, sum, and add bias.
    return output + bias

def main():
    # TODO: Create x with shape (1, 1, 4, 4).
    x = torch.tensor([[[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]]], dtype = float)
    # TODO: Create weight with shape (1, 1, 3, 3).
    w = torch.tensor([[[[1,1,1],[2,2,2],[3,3,3]]]], dtype = float)
    # TODO: Create bias with shape (1,).
    b = torch.tensor([1], dtype = float)
    # TODO: Run manual_conv2d.
    output = manual_conv2d(x,w,b)
    # TODO: Run F.conv2d.
    output_real = F.conv2d(x,w,b)
    # TODO: Print both outputs and their max absolute difference.
    print(f"manual: {output}, real: {output_real}")
    print("max abs diff:", torch.max(torch.abs(output - output_real)))
if __name__ == "__main__":
    main()


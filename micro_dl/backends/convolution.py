"""Vectorized concrete Runtime convolution kernels."""


def forward(xp, value, weight, bias, stride, padding):
    columns, out_h, out_w = _columns(xp, value, weight.shape[2:], stride, padding)
    batch, out_channels = value.shape[0], weight.shape[0]
    output = columns @ weight.reshape(out_channels, -1).T
    output = output.reshape(batch, out_h, out_w, out_channels).transpose(0, 3, 1, 2)
    if bias is not None:
        output += bias[None, :, None, None]
    return output


def backward(xp, value, weight, grad, stride, padding):
    columns, out_h, out_w = _columns(xp, value, weight.shape[2:], stride, padding)
    batch, channels, height, width = value.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    grad_rows = grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)
    grad_weight = (grad_rows.T @ columns).reshape(weight.shape)
    grad_columns = (grad_rows @ weight.reshape(out_channels, -1)).reshape(
        batch, out_h, out_w, channels, kernel_h, kernel_w
    )
    grad_padded = xp.zeros(
        (batch, channels, height + 2 * padding, width + 2 * padding), dtype=value.dtype
    )
    for kernel_row in range(kernel_h):
        for kernel_column in range(kernel_w):
            grad_padded[
                :, :, kernel_row:kernel_row + stride * out_h:stride,
                kernel_column:kernel_column + stride * out_w:stride,
            ] += grad_columns[:, :, :, :, kernel_row, kernel_column].transpose(0, 3, 1, 2)
    grad_value = grad_padded[:, :, padding:padding + height, padding:padding + width]
    grad_bias = xp.sum(grad, axis=(0, 2, 3))
    return grad_value, grad_weight, grad_bias


def _columns(xp, value, kernel, stride, padding):
    padded = xp.pad(value, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    batch, channels, height, width = value.shape
    kernel_h, kernel_w = kernel
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1
    shape = (batch, channels, out_h, out_w, kernel_h, kernel_w)
    base = padded.strides
    strides = (base[0], base[1], base[2] * stride, base[3] * stride, base[2], base[3])
    windows = xp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch * out_h * out_w, -1), out_h, out_w

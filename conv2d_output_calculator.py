import torch
import torch.nn as nn

def clean_input(input_):
    out = input_.split(' ')
    if len(out) == 1:
        out = int(out[0])
    else:
        out = tuple(int(f) for f in out)
    return out


while True:
    print('------------------------------------------------------------- \n')
    print('This script is to assist the calculate of the network output \n')
    print('input image size -> 28x28 ( Width x Height) \n')
    print('input filter size -> Filter size, Padding size and Stride number \n')
    print('The output size calculation is: (W - F + 2*P)/S + 1 \n')
    input_size = input('input the image size (e.g.: 224 224):')
    kernel_size = input('input kernel size: (e.g.: 3 or 2 4):')
    stride = input('input stride length: ')
    padding = input('input padding (e.g.: 2 or 1 3): ')

    input_size = clean_input(input_size)
    kernel_size = clean_input(kernel_size)
    stride = clean_input(stride) if stride else 1
    padding = clean_input(padding) if padding else 0
    print(f'input_size: {input_size},kernel_size: {kernel_size}, stride: {stride}, padding: {padding}')
    conv2d = nn.Conv2d(in_channels=1,
                       out_channels=1,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
    if isinstance(input_size,int):
        img = torch.randn(1,1,input_size,input_size)
    else:
        img = torch.randn(1,1,input_size[0],input_size[1])
    output = conv2d(img)
    print('output size: ',output.shape)

    sign = ['y','yes']
    yes = input("continue (y/n):")
    if not yes.lower() in sign:
        break 
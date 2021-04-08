import numpy as np

'''
numpy.pad(array, pad_width, mode='constant', **kwargs)
Parameters:
    pad_width{sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), … (before_N, after_N)) unique pad widths for each
        axis. ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all axes
'''

def pool2D_max(A, kernel_size, stride):
    w = (len(A) - kernel_size) // stride
    r=range(0,len(A),w)
    output = []

    for j in r:
        outputRow = []
        for i in r:
            windows = []
            for x in A[j:j+w]:
                # flatten the windows at the same time
                windows += x[i:i+w]
            max_windowsValue = max(windows)
            outputRow.append(max_windowsValue)
        output.append(outputRow)
    return output


def main():
    array = [[2, 9, 3, 8], [0, 1, 5, 5], [5, 7, 2, 6], [8, 8, 3, 6]]
    kernel_size = 2
    stride = 1
    output = pool2D_max(array, kernel_size, stride)
    print(output)


if __name__=="__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randint, rand

def plotInput(focused, unfocused):
    randInt = randint(0,focused.shape[0])
    
    focused = focused[randInt, :, :, 0]
    unfocused = unfocused[randInt, :, :, 0]
    vmin, vmax = calcMinMax(focused, unfocused)

    fig = plt.figure(figsize=(10, 10))
    
    ax1 = fig.add_subplot(1, 2, 1)
    p1 = ax1.imshow(focused, vmin = vmin, vmax = vmax, cmap='gray_r')
    plt.title('focused')

    ax2 = fig.add_subplot(1, 2, 2)
    p2 = ax2.imshow(unfocused, vmin = vmin, vmax = vmax, cmap='gray_r')
    plt.title('defocused')
    
    cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.4])
    fig.colorbar(p2,  cbar_ax)

    plt.show()
    
def calcMinMax(focused, unfocused):
    focused_values = np.reshape(focused, (focused.shape[0]*focused.shape[1],1))
    unfocused_values = np.reshape(unfocused, (unfocused.shape[0]*unfocused.shape[1],1))
    vmin = np.minimum(np.min(focused_values), np.min(unfocused_values))
    vmax = np.maximum(np.max(focused_values), np.max(unfocused_values))
    return vmin, vmax
    
def plotLoss(loss, save): 
    d_loss_true, d_loss_fake, g_loss  = loss[0], loss[1], loss[2]
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_true,'r')
    plt.plot(d_loss_fake,'b')
    plt.plot(g_loss,'k')
    fontsize = 12
    plt.legend(['Discriminator Loss on True', 'Discriminator Loss on Fake', 'Generator Loss'], loc='upper right', fontsize=fontsize)
    plt.ylabel("Loss", fontname="Arial", fontsize=fontsize)
    plt.xlabel("Epochs", fontname="Arial", fontsize=fontsize)
    plt.title("Loss Vs. Epochs", fontname="Arial", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    if save == 1:
        plt.savefig(r"loss.png")
def plotGen(samples, n_samples):
    # plot images
    fig = plt.figure(figsize=(15, 15))
    for i in range(n_samples * n_samples):
        # define subplot
        fig.add_subplot(n_samples, n_samples, 1 + i)
        #plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(samples[i, :, :, 0], cmap='gray_r')
    plt.show()
def getModelInput(data):
    real = data[1,:,:,:]
    cond = data[0,:,:,:]
    real = np.expand_dims(real, axis = -1)
    cond = np.expand_dims(cond, axis = -1)
    return real, cond

def decimateInput(data):
    num_pixels = 128
    skip_pixels = 1
    start_pixel = 1
    #decimate the input
    data = data[:,:,:num_pixels,:num_pixels]
    data = data[:,:,start_pixel::skip_pixels+1,start_pixel::skip_pixels+1]
    return data

def dataProcess(data):
    #decimate the input
    data = decimateInput(data)
    #seperate focused from unfocused image (real and conditional)
    real, cond = getModelInput(data)
    #standardize the data
    real, cond = standardizeInput(real, cond)
    print('input_real shape: ', real.shape)
    print('input_cond shape: ', cond.shape)
    return real, cond

def standardize(x):
    xStretch = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3],1))
    mu_x = np.mean(xStretch)
    std_x = np.std(xStretch)
    xStretch_std = (xStretch - mu_x)/std_x
    x = np.reshape(xStretch_std, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    return x
def standardizeInput(real, cond):
    real = standardize(real)
    cond = standardize(cond)
    return real, cond
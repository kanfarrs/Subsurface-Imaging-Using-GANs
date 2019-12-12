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
    p1 = ax1.imshow(focused, cmap='gray_r')

    plt.title('focused')

    ax2 = fig.add_subplot(1, 2, 2)
    p2 = ax2.imshow(unfocused, cmap='gray_r')
    p2 = ax1.imshow(focused, vmin = vmin, vmax = vmax, cmap='gray_r')
    plt.title('defocused')
    
    cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.4])
    fig.colorbar(p2,  cbar_ax)

    plt.show()
    
def plotCompare(x_real, x_fake, input_cond, n_samples):
    # plot images
    fig = plt.figure(figsize=(15, 15))
    for i in range(n_samples + 2):
        fig.add_subplot(n_samples + 2, n_samples + 2, (i*(n_samples+2))+1)
        plt.axis('off')
        plt.imshow(input_cond[i,:, :,0], cmap='gray_r')
        for j in range(n_samples):
            idx = (n_samples+2)*i + j + 2
            fig.add_subplot(n_samples + 2, n_samples + 2, idx)
            plt.axis('off')
            plt.imshow(x_fake[i,j, :, :, 0], cmap='gray_r')
        fig.add_subplot(n_samples + 2, n_samples + 2, idx + 1)
        plt.imshow(x_real[i, :, :, 0], cmap='gray_r')   
        plt.axis('off')
    plt.show()
    
def plotCompareOne(x, y_fake, y_real): #make into loop
    # plot images
    randInt = randint(0,y_fake.shape[0])
    
    y_real = y_real[randInt, :, :, 0]
    y_real = y_real/np.max(abs(y_real))
    x = x[randInt, :, :, 0]
    x = x/np.max(abs(x))
    y_fake = y_fake[randInt, :, :, 0]
    y_fake = y_fake/np.max(abs(y_fake))

    fontsize = 12
    #vmin, vmax = calcMinMax_compare(y_real, x, y_fake)

    fig = plt.figure(figsize=(10, 10))
    
    ax1 = fig.add_subplot(1, 3, 1)
    p1 = ax1.imshow(x, cmap='gray')
    #p1 = ax1.imshow(y_real, vmin = vmin*0.99, vmax = vmax*0.99, cmap='gray_r')

    plt.title('Defocused',fontsize = fontsize)

    ax2 = fig.add_subplot(1, 3, 2)
    p2 = ax2.imshow(y_fake, cmap='gray')
    #p2 = ax2.imshow(x, vmin = vmin*0.99, vmax = vmax*0.99, cmap='gray_r')
    plt.title('Generated', fontsize = fontsize)

    ax3 = fig.add_subplot(1, 3, 3)
    p3 = ax3.imshow(y_real, cmap='gray')

    #p3 = ax3.imshow(y_fake, vmin = vmin*0.99, vmax = vmax*0.99, cmap='gray_r')
    plt.title('Focused', fontsize = fontsize)
    
    cbar_ax = fig.add_axes([0.95, 0.3, 0.05, 0.4])
    fig.colorbar(p1,  cbar_ax)

    plt.show()
    
def calcMinMax(focused, unfocused):
    focused_values = np.reshape(focused, (focused.shape[0]*focused.shape[1],1))
    unfocused_values = np.reshape(unfocused, (unfocused.shape[0]*unfocused.shape[1],1))
    vmin = np.minimum(np.min(focused_values), np.min(unfocused_values))
    vmax = np.maximum(np.max(focused_values), np.max(unfocused_values))
    return vmin, vmax

def calcMinMax_compare(focused, unfocused, generated): #combined with function above in a loop or optional arg.
    focused_values = np.reshape(focused, (focused.shape[0]*focused.shape[1],1))
    unfocused_values = np.reshape(unfocused, (unfocused.shape[0]*unfocused.shape[1],1))
    generated_values = np.reshape(generated, (generated.shape[0]*generated.shape[1],1))

    vmin = np.minimum(np.min(focused_values), np.min(unfocused_values))
    vmin = np.minimum(vmin, np.min(generated_values))
    vmax = np.maximum(np.max(focused_values), np.max(unfocused_values))
    vmax = np.maximum(vmax, np.min(generated_values))
    return vmin, vmax
    
def plotLoss(loss, save): 
    d_loss_true, d_loss_fake, g_loss  = loss[0], loss[1], loss[2]
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_true,'r')
    plt.plot(d_loss_fake,'b')
    plt.plot(g_loss,'k')

    #plt.plot(d_loss_true/np.amax(d_loss_true),'r')
    #plt.plot(d_loss_fake/np.amax(d_loss_fake),'b')
    #plt.plot(g_loss/np.max(g_loss),'k')
    fontsize = 15
    plt.legend(['Discriminator Loss on True', 'Discriminator Loss on Fake', 'Generator Loss'], loc='upper right', fontsize=fontsize)
    plt.ylabel("Loss", fontname="Arial", fontsize=fontsize)
    plt.xlabel("Epochs", fontname="Arial", fontsize=fontsize)
    plt.title("Loss Vs. Epochs", fontname="Arial", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    if save == 1:
        plt.savefig(r"loss.png")

def plotLossP2P(loss, save): 
    d_loss_true, d_loss_fake, mse_loss, g_loss  = loss[0], loss[1], loss[2], loss[3]
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_true,'r')
    plt.plot(d_loss_fake,'b')
    plt.plot(mse_loss,'k')
    plt.plot(g_loss,'g')

    fontsize = 12
    plt.legend(['Discriminator Loss on True', 'Discriminator Loss on Fake', 'MSE Loss','Generator Loss'], loc='upper right', fontsize=fontsize)
    plt.ylabel("Loss", fontname="Arial", fontsize=fontsize)
    plt.xlabel("Epochs", fontname="Arial", fontsize=fontsize)
    plt.title("Loss Vs. Epochs", fontname="Arial", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    if save == 1:
        plt.savefig(r"loss.png")

def plotLossCycleGAN(loss, save): 
    f_loss_epoch, g_loss_epoch, dG_loss_fake_epoch, dG_loss_real_epoch, dF_loss_fake_epoch, dF_loss_real_epoch  = loss[0], loss[1], loss[2], loss[3], loss[4], loss[5]
    #plt.figure(figsize=(10, 5))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    fontsize = 12

    ax1.plot(dG_loss_fake_epoch,'b')
    ax1.plot(dG_loss_real_epoch,'r')
    ax1.plot(dF_loss_fake_epoch,'--b')
    ax1.plot(dF_loss_real_epoch,'--r')
    ax1.legend(['dG Loss on fake', 'dG Loss on true', 'dF Loss on fake', 'dF Loss on true'], loc='upper right', fontsize=fontsize)

    ax2.plot(g_loss_epoch,'k', linewidth=2)
    ax2.plot(f_loss_epoch,'--k', linewidth=2)
    ax2.legend(['g_loss', 'f_loss'], loc='upper left', fontsize=fontsize)


    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Discriminator Loss', color='g', fontsize = fontsize)
    ax2.set_ylabel('Generator Loss', color='b', fontsize = fontsize)

    plt.show()
    '''
    plt.plot(dG_loss_fake_epoch,'b')
    plt.plot(dG_loss_real_epoch,'r')
    plt.plot(g_loss_epoch,'k', linewidth=2)

    plt.plot(dF_loss_fake_epoch,'--b')
    plt.plot(dF_loss_real_epoch,'--r')
    plt.plot(f_loss_epoch,'--k', linewidth=2)

    fontsize = 12
    plt.legend(['dG Loss on fake', 'dG Loss on true', 'g Loss', 'dF Loss on fake', 'dF Loss on true', 'f Loss'], loc='upper right', fontsize=fontsize)
    plt.ylabel("Loss", fontname="Arial", fontsize=fontsize)
    plt.xlabel("Epochs", fontname="Arial", fontsize=fontsize)
    plt.title("Loss Vs. Epochs", fontname="Arial", fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    '''
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
        plt.imshow(samples[i, :, :, 0], cmap='gray')
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
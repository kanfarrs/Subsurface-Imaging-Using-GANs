import numpy as np
import matplotlib.pyplot as plt

class plotGAN:
    def plotLoss(loss, save): 
        d_loss_true, d_loss_fake, g_loss  = loss[0], loss[1], loss[2]
        plt.figure(figsize=(15, 15))
        plt.plot(d_loss_true,'r')
        plt.plot(d_loss_fake,'b')
        plt.plot(g_loss,'k')
        fontsize = 20
        plt.legend(['Discriminator Loss on True', 'Discriminator Loss on Fake', 'Generator Loss'], loc='upper right', fontsize=fontsize)
        ax.set_ylabel("Loss", fontname="Arial", fontsize=fontsize)
        ax.set_xlabel("Epochs", fontname="Arial", fontsize=fontsize)
        ax.set_title("Loss Vs. Epochs", fontname="Arial", fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
    if save == 1:
        plt.savefig(r"loss.png")
    def plotGen(examples, n):
        # plot images
        fig = plt.figure(figsize=(15, 15))
        for i in range(n * n):
            # define subplot
            fig.add_subplot(n, n, 1 + i)
            #plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        plt.show()
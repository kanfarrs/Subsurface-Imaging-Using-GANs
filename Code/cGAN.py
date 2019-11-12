# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:50:46 2019

@author: kanfar
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt
from numpy import expand_dims, zeros, ones
from numpy.random import randn, randint
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, Concatenate
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Dropout, LeakyReLU

class cGAN:
    def __init__(self, input_dim1, input_dim2, input_dim3, latent_size):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.input_dim3 = input_dim3
        self.latent_size = latent_size
    def discriminator(self):
        #conditional input
        input_shape = (self.input_dim1, self.input_dim2, self.input_dim3)
        input_cond = Input(shape = input_shape)
        #generator output
        input_x = Input(shape = input_shape)
        merge = Concatenate()([input_x, input_cond])
        #downsample
        out = Conv2D(32, (3,3), strides=(2,2), padding='same')(merge)
        out = LeakyReLU(alpha=0.2)(out)
        out = Conv2D(32, (3,3), strides=(2,2), padding='same')(out)
        out = LeakyReLU(alpha=0.2)(out)
        out = Flatten()(out)
        out = Dropout(0.5)(out)
        y = Dense(1, activation='sigmoid')(out)
        # define model
        model = Model([input_x, input_cond], y)
        # compile model
        opt = Adam(lr=0.00005) #0.0002 and beta_1 0.5
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    def generator(self):
        #losing one pixel, figure out later
        image_dim = self.input_dim1
        latent_shape = self.latent_size
        cond_shape = (image_dim, image_dim, self.input_dim3)
        
        input_latent = Input(shape = (latent_shape,))
        num_nodes = image_dim * image_dim
        latent = Dense(num_nodes)(input_latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Reshape((image_dim,image_dim,1))(latent)
        
        input_cond = Input(shape = cond_shape)
        cond = input_cond
        
        merge = Concatenate()([latent,cond])
        
        # upsample to 14x14
        out = Conv2D(32, (4,4), strides=(1,1), padding='same')(merge)
        out = LeakyReLU(alpha=0.2)(out)
        # upsample to 28x28
        out = Conv2D(32, (4,4), strides=(1,1), padding='same')(out)
        out = LeakyReLU(alpha=0.2)(out)
        
        out = Conv2D(32, (4,4), strides=(1,1), padding='same')(out)
        out = LeakyReLU(alpha=0.2)(out)
        # output
        x = Conv2D(1, (4,4), strides=(1,1), activation='tanh', padding='same')(out) #something key that I don't understand
        # define model
        model = Model([input_latent, input_cond], x)
        return model
    def combined(self, g_model, d_model):
        #model comprised of two models
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        input_latent, input_cond = g_model.input #defining the tensors in a short way: this is saying the input to this model is the same size as input to g_model
        # get image output from the generator model
        x = g_model.output
        #can I do x = g_model([input_latent, input_cond]) instead of the above?
        # connect image output and label input from generator as inputs to discriminator
        y = d_model([x, input_cond]) #why this needs to be connected but not the above???? does the first output take model input as default??????? test this
        # define gan model as taking noise and label and outputting a classification
        model = Model([input_latent, input_cond], y)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
    def generate_real_samples(self, focused, defocused, n_samples):
        idx = randint(0, focused.shape[0], n_samples)
        x_real, input_cond = focused[idx,:,:,:], defocused[idx,:,:,:] 
        y_real = ones((n_samples,1))
        return [x_real, input_cond], y_real
    
    def generate_latent(self, latent_size, n_samples):
        #generate points in teh latent space
        total_latent = randn(latent_size*n_samples)
        input_z = total_latent.reshape(n_samples, latent_size) 
        return input_z
    def generate_fake_samples(self, generator, defocused, latent_dim, n_samples):
        idx = randint(0, defocused.shape[0], n_samples)
        input_cond = defocused[idx,:,:,:] ##### should last be zero or :?
        input_z = self.generate_latent(latent_dim, n_samples)
        # predict outputs
        x_fake = generator.predict([input_z, input_cond])
        # create class labels
        y_fake = zeros((n_samples, 1))
        return [x_fake, input_cond], y_fake
    def generate_gan_input(self, defocused, latent_dim, n_samples):
        #defocused = data[1,:,:,:]
        #defocused = np.expand_dims(input_cond, axis = -1)
        idx = randint(0, defocused.shape[0], n_samples)
        input_cond = defocused[idx,:,:,:] ##### should last be zero or :?
        input_z = self.generate_latent(latent_dim, n_samples)
        # create class labels
        y_gan = ones((n_samples, 1))
        return [input_z, input_cond], y_gan

    def train(self, g_model, d_model, gan_model, real, input_cond, latent_dim, n_epochs, n_batch):
        bat_per_epo = int(real.shape[0] / n_batch) #check
        half_batch = int(n_batch / 2)
        g_loss = np.zeros(n_epochs)
        d_loss_real = np.zeros(n_epochs)
        d_loss_fake = np.zeros(n_epochs)
        # manually enumerate epochs
        for i in range(n_epochs):
            start = timeit.default_timer()
            # enumerate batches over the training set
            print('================== Epoch %d ==================\n' % (i+1))
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [x_real, input_cond_real], y_real = self.generate_real_samples(real, input_cond, half_batch)
                # update discriminator model weights
                d_loss_real[i], _ = d_model.train_on_batch([x_real, input_cond_real], y_real)
                # generate 'fake' examples
                [x_fake, input_cond_fake], y_fake = self.generate_fake_samples(g_model, input_cond, latent_dim, half_batch)
                # update discriminator model weights
                d_loss_fake[i], _ = d_model.train_on_batch([x_fake, input_cond_fake], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, input_cond_gan], y_gan = self.generate_gan_input(input_cond, latent_dim, n_batch)
                # update the generator via the discriminator's error
                g_loss[i] = gan_model.train_on_batch([z_input, input_cond_gan], y_gan)
                # summarize loss on this batch
                print('Completed: %.f' % np.divide((j+1)*100,bat_per_epo) +'%')
            print('Epoch %d:: d_loss_real = %.3f, d_loss_fake = %.3f g_loss = %.3f' %
                    (i+1, d_loss_real[i], d_loss_fake[i], g_loss[i]) + '\n')
            stop = timeit.default_timer()
            print('Time: %.2f min' % ((stop - start)/60)) 
        # save the generator model
        g_model.save('cgan_regular_50.h5') #save somewhere
        # save loss history
        loss = np.array([d_loss_real, d_loss_fake, g_loss])
        np.save('cgan_regular_50', loss)
    def save_plot(self, examples, n):
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
        
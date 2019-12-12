"""
Discription: Implementation of cycleGAN in Keras
File:model.py
Packages: requires keras and related dependencies.

Author: Rayan Kanfar (kanfar@stanford.edu)
reference: https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
Date: 11/22/2019
"""
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization #keras-contrib library can be installed using: pip install git+https://www.github.com/keras-team/keras-contrib.git
import numpy as np
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.initializers import RandomNormal
from numpy.random import randn, randint
from numpy import expand_dims, zeros, ones, asarray, random
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Input
from tensorflow.python.keras.layers import Conv2D, Flatten, Dropout, Dense, Reshape
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import BatchNormalization
import timeit
import os

"""
to do:
- implement receiptive field function (will be helpful later)

"""


class pix2pix:
    def __init__(self, input_shape, n_filters):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.n_resblocks = 9


# define the discriminator model
# desired
    def discriminator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        cond_image = Input(shape= self.input_shape)
        # target image input
        y_image = Input(shape= self.input_shape)
        # concatenate images channel-wise
        merged = Concatenate()([cond_image, y_image])
        d = Conv2D(self.n_filters, (4,4), strides=(3,3), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(4*self.n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([cond_image, y_image], patch_out)
        # compile model
        opt = Adam(lr=0.0001, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
 

# define the standalone generator model
    def generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=self.input_shape)
        # c7s1-64
        g = Conv2D(self.n_filters, (11,11), padding='same', kernel_initializer=init)(in_image)
        g = Activation('tanh')(g)

        g = Conv2D(2*self.n_filters, (7,7), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2D(4*self.n_filters, (5,5), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2D(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2D(4*self.n_filters, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2D(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2D(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        #for _ in range(self.n_resblocks):
        #    g = self.resnet_block(g)

        g = Conv2DTranspose(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2DTranspose(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2DTranspose(4*self.n_filters, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2DTranspose(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2DTranspose(4*self.n_filters, (5,5), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        g = Conv2DTranspose(2*self.n_filters, (7,7), strides=(2,2), padding='same', kernel_initializer=init)(g)
        g = Activation('tanh')(g)

        # c7s1-3
        g = Conv2D(1, (11,11), padding='same', kernel_initializer=init)(g)
        #g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model

    def compositeGAN(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # define the source image
        cond_image = Input(shape=self.input_shape)
        # connect the source image to the generator input
        g_out = g_model(cond_image)
        # connect the source input and generator output to the discriminator input
        d_out = d_model([cond_image, g_out])
        # src image as input, generated image and classification output
        model = Model(cond_image, d_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy'], optimizer=opt) ##############wtf
        return model

    def mseG(self, g_model):
        cond_image = Input(shape = self.input_shape)
        # connect the source image to the generator input
        g_out = g_model(cond_image)
        # src image as input, generated image and classification output
        model = Model(cond_image, g_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=[ 'mse'], optimizer=opt) ##############wtf
        return model


# select a batch of random samples, returns images and target
    def generate_real_samples(self, defocused, focused, n_samples, patch_shape):
        # choose random instances
        ix = randint(0, focused.shape[0], n_samples)
        # retrieve selected images
        cond_image, real_image = defocused[ix], focused[ix]
        # generate 'real' class labels (1)
        y_real = ones((n_samples, patch_shape, patch_shape, 1))
        return [cond_image, real_image], y_real

# generate a batch of images, returns images and targets
    def generate_fake_samples(self, g_model, samples, patch_shape):
        # generate fake instance
        fake_image = g_model.predict(samples)
        # create 'fake' class labels (0)
        y_fake = zeros((len(fake_image), patch_shape, patch_shape, 1))
        return fake_image, y_fake

    def initialLoss(self, bat_per_epo, n_epochs):
        g_loss = np.zeros(bat_per_epo)
        d_loss_fake = np.zeros(bat_per_epo)
        d_loss_real = np.zeros(bat_per_epo)
        mse_loss = np.zeros(bat_per_epo)

        mse_loss_epoch = np.zeros(n_epochs)
        g_loss_epoch = np.zeros(n_epochs)
        d_loss_fake_epoch = np.zeros(n_epochs)
        d_loss_real_epoch = np.zeros(n_epochs)
        return g_loss, d_loss_fake, d_loss_real, g_loss_epoch, d_loss_fake_epoch, d_loss_real_epoch, mse_loss, mse_loss_epoch

# train cyclegan models
    def train(self, d_model, g_model, mseG, gan_model, dataX, dataY, batch_size, n_epochs, save):
        print('wtf is going on')
        helper = 10 #what's a more efficient way to do this...
        plateau = 0 #is there a difference between two loss terms or one loss term cascaded?
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(dataX) / batch_size) #assumes size of dataX = dataY
        #initializations
        g_loss, d_loss_fake, d_loss_real, g_loss_epoch, d_loss_fake_epoch, d_loss_real_epoch, mse_loss, mse_loss_epoch = self.initialLoss(bat_per_epo, n_epochs)
        # determine the output square shape of the discriminator
        n_patch = d_model.output_shape[1] #
        startTotal = timeit.default_timer()
        for i in range(n_epochs):
            start = timeit.default_timer()
            if helper < 0: #I was trying to avoid this...but maybe once per epoch isn't too bad
                helper = 0
            for j in range(bat_per_epo):
                mse_loss_temp = np.zeros(abs(helper) + plateau)
                [cond_image, real_image], y_real = self.generate_real_samples(dataX, dataY, batch_size, n_patch)
                fake_image, y_fake = self.generate_fake_samples(g_model, cond_image, n_patch)
                # update discriminator for real samples
                d_loss_real[j] = d_model.train_on_batch([cond_image, real_image], y_real)
                # update discriminator for generated samples
                d_loss_fake[j] = d_model.train_on_batch([cond_image, fake_image], y_fake)
                # select a batch of real samples
                for k in range(abs(helper) + plateau): #could instead do while
                    [cond_image_temp, real_image_temp], y_real = self.generate_real_samples(dataX, dataY, batch_size, n_patch)
                    mse_loss_temp[k] = mseG.train_on_batch(cond_image_temp,real_image_temp)
                    print('Epoch %d:: helper = %.3f, mse_loss = %.3f' %
                        (i+1, helper, mse_loss_temp[k]) + '\n')
                mse_loss[j] = np.mean(mse_loss_temp)
                # generate a batch of fake samples
                # update the generator
                g_loss[j] = gan_model.train_on_batch(cond_image, y_real)
                print('Completed: %.f' % np.divide((j+1)*100,bat_per_epo) +'%')
                # summarize performance
            helper = helper - 1
            mse_loss_epoch[i] = np.mean(mse_loss)
            g_loss_epoch[i] = np.mean(g_loss)
            d_loss_fake_epoch[i] = np.mean(d_loss_fake)
            d_loss_real_epoch[i] = np.mean(d_loss_real)
            stop = timeit.default_timer()
            print('Epoch %d:: d_loss_real = %.3f, d_loss_fake = %.3f, mse_loss = %.3f, g_loss = %.3f' %
                (i+1, d_loss_real_epoch[i], d_loss_fake_epoch[i], mse_loss_epoch[i], g_loss_epoch[i]) + '\n')
            print('Time: %.2f min' % ((stop - start)/60))
            if i == 30:
                os.mkdir('./models/pix2pix/' + save)
                ###### save models ######
                d_model.save('./models/pix2pix/30' + save + '/' + 'd' + '.h5') 
                g_model.save('./models/pix2pix/30' + save + '/' + 'g' + '.h5') 
                gan_model.save('./models/pix2pix/30' + save + '/' + 'composite' + '.h5') 
                stopTotal = timeit.default_timer()
        print('Total Time: %.2f min' % ((stopTotal - startTotal)/60)) 
        #create directory to save model 
        os.mkdir('./models/pix2pix/' + save)
        ###### save models ######
        d_model.save('./models/pix2pix/' + save + '/' + 'd' + '.h5') 
        g_model.save('./models/pix2pix/' + save + '/' + 'g' + '.h5') 
        gan_model.save('./models/pix2pix/' + save + '/' + 'composite' + '.h5') 

        loss = np.array([d_loss_real_epoch, d_loss_fake_epoch, mse_loss_epoch, g_loss_epoch])
        np.save('./models/pix2pix/' + save + '/loss', loss)

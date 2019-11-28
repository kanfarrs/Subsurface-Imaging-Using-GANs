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


class cycleGAN:
    def __init__(self, input_shape, n_filters):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.n_resblocks = 9


# define the discriminator model
# desired
    def discriminator(self):
        #generator output
        # input_x = Input(shape = self.input_shape)
        # out = Conv2D(32, (3,3), strides=(2,2), padding='same')(input_x)
        # out = LeakyReLU(alpha=0.2)(out)
        # out = Conv2D(32, (3,3), strides=(2,2), padding='same')(out)
        # out = LeakyReLU(alpha=0.2)(out)
        # out = Flatten()(out)
        # out = Dropout(0.5)(out)
        # y = Dense(1, activation='sigmoid')(out)
        # #y = Reshape((1,1,1))(y)
        # # define model
        # model = Model(input_x, y)
        # # compile model
        # #opt = Adam(lr=0.0002) #0.0002 and beta_1 0.5
        # model.compile(loss='binary_crossentropy', optimizer = 'Adam')
        # return model
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_image = Input(shape=self.input_shape)
        # C64
        d = Conv2D(self.n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(2*self.n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        #d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(2*self.n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        #d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(4*self.n_filters, (4,4), padding='same', kernel_initializer=init)(d)
        #d = InstanceNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model(in_image, patch_out)
        # compile model
        opt = Adam(lr=0.0001, beta_1=0.5) #0.0002 and beta_1 0.5

        model.compile(loss='mse', optimizer='Adam', loss_weights=[0.25]) #Adam(lr=0.0002, beta_1=0.5)
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

    # generator a resnet block
    def resnet_block(self, input_layer):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # first layer convolutional layer
        g = Conv2D(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        #g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # second convolutional layer
        g = Conv2D(4*self.n_filters, (3,3), padding='same', kernel_initializer=init)(g)
        #g = InstanceNormalization(axis=-1)(g)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g

# define a composite model for updating generators by adversarial and cycle loss
    def composite_model(self, g_model_1, d_model, g_model_2, image_shape):
        # the loss includes 4 terms for each generator: GAN, optimize over x, optimize over y, identity y (all ways to use G given the unpaired data)
        # ensure the model we're updating is trainable
        g_model_1.trainable = True
        # mark discriminator as not trainable
        d_model.trainable = False
        # mark other generator model as not trainable
        g_model_2.trainable = False
        # 1) reagular GAN pass: genenerator -> discriminator  (D(G(x))), optimize to be real
        input_gen = Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)
        # 2) forward cycle consistency step (F(G(x)), optimize to be x (form of identity)
        output_f = g_model_2(gen1_out)
        # 3) backward cycle consistency step (G(F(y))), optimize to be y (form of identity)
        input_id = Input(shape=image_shape)
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)
        # 4) generator identity pass (G(x)), optimize to be y (identity)
        output_id = g_model_1(input_id)
        # define model graph
        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        # define optimization algorithm configuration
        #opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[4, 1, 1, 1], optimizer='Adam')
        return model

# select a batch of random samples, returns images and target
    def generate_real_samples(self, dataset, n_samples, patch_shape):
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        #y = ones((n_samples, 1))
        return X, y

# generate a batch of images, returns images and targets
    def generate_fake_samples(self, g_model, dataset, patch_shape):
        # generate fake instance
        X = g_model.predict(dataset)
        # create 'fake' class labels (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        #y = ones((len(X), 1))
        return X, y

# update image pool for fake images
    def update_image_pool(self, pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif random.random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return asarray(selected)
    def initialLoss(self, bat_per_epo, n_epochs):
        f_loss = np.zeros(bat_per_epo)
        g_loss = np.zeros(bat_per_epo)
        dG_loss_fake = np.zeros(bat_per_epo)
        dG_loss_real = np.zeros(bat_per_epo)
        dF_loss_fake = np.zeros(bat_per_epo)
        dF_loss_real = np.zeros(bat_per_epo)

        f_loss_epoch = np.zeros(n_epochs)
        g_loss_epoch = np.zeros(n_epochs)
        dG_loss_fake_epoch = np.zeros(n_epochs)
        dG_loss_real_epoch = np.zeros(n_epochs)
        dF_loss_fake_epoch = np.zeros(n_epochs)
        dF_loss_real_epoch = np.zeros(n_epochs)
        return f_loss, g_loss, dG_loss_fake, dG_loss_real, dF_loss_fake, dF_loss_real, f_loss_epoch, g_loss_epoch, dG_loss_fake_epoch, dG_loss_real_epoch, dF_loss_fake_epoch, dF_loss_real_epoch

# train cyclegan models
    def train(self, d_model_G, d_model_F, g_model_G, g_model_F, c_model_GtoF, c_model_FtoG, dataX, dataY, batch_size, n_epochs, save):
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(dataX) / batch_size) #assumes size of dataX = dataY
        #initializations
        f_loss, g_loss, dG_loss_fake, dG_loss_real, dF_loss_fake, dF_loss_real, f_loss_epoch, g_loss_epoch, dG_loss_fake_epoch, dG_loss_real_epoch, dF_loss_fake_epoch, dF_loss_real_epoch = self.initialLoss(bat_per_epo, n_epochs)
        # determine the output square shape of the discriminator
        n_patch = d_model_G.output_shape[1] #
        # prepare image pool for fakes
        poolG, poolF = list(), list()
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        startTotal = timeit.default_timer()
        for i in range(n_epochs):
            start = timeit.default_timer()
            for j in range(bat_per_epo):
                # select a batch of real samples
                real_Y, label_realG = self.generate_real_samples(dataY, batch_size, n_patch)
                real_X, label_realF = self.generate_real_samples(dataX, batch_size, n_patch)
                # generate a batch of fake samples
                fake_Y, label_fakeG = self.generate_fake_samples(g_model_G, real_X, n_patch)
                fake_X, label_fakeF = self.generate_fake_samples(g_model_F, real_Y, n_patch)
                # update fakes from pool
                fake_Y = self.update_image_pool(poolG, fake_Y)
                fake_X = self.update_image_pool(poolF, fake_X)
                # update generator X->Y via adversarial and cycle loss
                g_loss[j], _, _, _, _ = c_model_GtoF.train_on_batch([real_X, real_Y], [label_realG, real_Y, real_X, real_Y])
                # update discriminator for Y -> [real/fake]

                dG_loss_real[j] = d_model_G.train_on_batch(real_Y, label_realG)
                dG_loss_fake[j] = d_model_G.train_on_batch(fake_Y, label_fakeG)
                # update generator Y-> X via adversarial and cycle loss
                f_loss[j], _, _, _, _  = c_model_FtoG.train_on_batch([real_Y, real_X], [label_realF, real_X, real_Y, real_X])
                # update discriminator for X -> [real/fake]
                dF_loss_real[j] = d_model_F.train_on_batch(real_X, label_realF)
                dF_loss_fake[j] = d_model_F.train_on_batch(fake_X, label_fakeF)
                # summarize performance
                print('Completed: %.f' % np.divide((j+1)*100,bat_per_epo) +'%')
            f_loss_epoch[i] = np.mean(f_loss)
            g_loss_epoch[i] = np.mean(g_loss)
            dG_loss_fake_epoch[i] = np.mean(dG_loss_fake)
            dG_loss_real_epoch[i] = np.mean(dG_loss_real)
            dF_loss_fake_epoch[i] = np.mean(dF_loss_fake)
            dF_loss_real_epoch[i] = np.mean(dF_loss_real)
            stop = timeit.default_timer()
            print('Epoch %d:: dG_loss_real = %.3f, dG_loss_fake = %.3f dF_loss_real = %.3f, dF_loss_fake = %.3f g_loss = %.3f, f_loss = %.3f' %
                (i+1, dG_loss_real_epoch[i], dG_loss_fake_epoch[i], dF_loss_real_epoch[i], dG_loss_fake_epoch[i], g_loss_epoch[i], f_loss_epoch[i]) + '\n')
            print('Time: %.2f min' % ((stop - start)/60)) 
        stopTotal = timeit.default_timer()
        print('Total Time: %.2f min' % ((stopTotal - startTotal)/60)) 
        #create directory to save model 
        os.mkdir('./models/cycleGAN/' + save)
        ###### save models ######
        d_model_G.save('./models/cycleGAN/' + save + '/' + 'dG' + '.h5') 
        d_model_F.save('./models/cycleGAN/' + save + '/' + 'dF' + '.h5') 
        g_model_G.save('./models/cycleGAN/' + save + '/' + 'G' + '.h5') 
        g_model_F.save('./models/cycleGAN/' + save + '/' + 'F' + '.h5') 
        c_model_GtoF.save('./models/cycleGAN/' + save + '/' + 'GtoF' + '.h5')
        c_model_FtoG.save('./models/cycleGAN/' + save + '/' + 'FtoG' + '.h5')
        # save loss history
        loss = np.array([f_loss_epoch, g_loss_epoch, dG_loss_fake_epoch, dG_loss_real_epoch, dF_loss_fake_epoch, dF_loss_real_epoch])
        np.save('./models/cycleGAN/' + save + '/loss', loss)

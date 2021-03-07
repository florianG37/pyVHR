##
## Importing libraries
##

#Tensorflow/KERAS
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import np_utils

# Numpy / Matplotlib / OpenCV / Scipy / Copy / ConfigParser
import numpy as np
import scipy.io
import scipy.stats as sp
import cv2
from copy import copy
import os
import configparser

#pyVHR
from pyVHR.signals.video import Video
from .base import VHRMethod

class MAP_3DCNN(VHRMethod):
    
    '''

    Method based on the following work : 
    title : "3D Convolutional Neural Networks for Remote Pulse Rate Measurement 
    and Mapping from Facial Video"
    authors : Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui
    url : https://www.mdpi.com/2076-3417/9/20/4364

    Implemented & adapted by GIGOT Florian (Student at Polytech'Tours)

    '''

    methodName = 'MAP_3DCNN'
    
    def __init__(self, **kwargs):
        self.x_step = int(kwargs['xstep']) 
        self.y_step = int(kwargs['ystep'])
        self.modelFilename = str(kwargs['modelFilename'])   
        super(MAP_3DCNN, self).__init__(**kwargs)

    ##--------------------------------------------------------------------------------
    #
    # apply(self, faces) : MAIN
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # faces : Video sequence submitted to the prediction (including the subject's face)
    #
    #---------------------------------------------------------------------------------
    # 
    # return : Estimation of the bpm associated with the video sequence
    #
    ##--------------------------------------------------------------------------------     
    def apply(self, faces):
        #load model 
        model, freq_bpm = self.loadmodel()

        # manage variation in the size of inputs
        faces = self.checkNbFrames(faces, model)

        #extract Green channel or Black & white channel
        frames_one_channel = self.convert_video_to_table(faces, model)
    
        prediction = self.formating_data_test(model, frames_one_channel, freq_bpm)
    
        # get bpm
        bpm = self.get_bpm(prediction, freq_bpm)

        # memory management
        del model
        del freq_bpm
        del faces
        del frames_one_channel
        del prediction
        
        return np.asarray([bpm])

    ##--------------------------------------------------------------------------------
    #
    # get_bpm(self,prediction, freq_bpm) : Applying the formula to transform the 
    #   prediction result into a value representing the estimated heart rate (BPM)
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # prediction : array including the addition of all predictions
    #
    # freq_bpm : array containing the set of classes (representing each bpm) known 
    #   by the model
    #
    #---------------------------------------------------------------------------------
    # 
    # return : bpm value calculated
    #
    ##-------------------------------------------------------------------------------- 
    def get_bpm(self,prediction, freq_bpm): 
        nb_bins = 0
        score = 0
        for i in range(len(prediction)-1):
            nb_bins += prediction[i]
            score += freq_bpm[i] * prediction[i]
        
        bpm = score / nb_bins
    
        return bpm


    ##--------------------------------------------------------------------------------
    #
    # loadmodel(self) : Loading the model trained
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    #---------------------------------------------------------------------------------
    # 
    # return : 
    #
    # model : the model trained to make predictions
    #
    # freq_bpm : array containing the set of classes (representing each bpm) known 
    #   by the model
    #
    ##-------------------------------------------------------------------------------- 
    def loadmodel(self):
        model = model_from_json(open(f'{self.modelFilename}/model_conv3D.json').read())
        model.load_weights(f'{self.modelFilename}/weights_conv3D.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # define the frequencies // output dimension (number of classes used during training)
        freq_bpm = np.linspace(55, 240, num=model.output_shape[1]-1)
        freq_bpm = np.append(freq_bpm, -1)     # noise class
        return model, freq_bpm


    ##--------------------------------------------------------------------------------
    #
    # convert_video_to_table(self, faces ,model) : Converting videoframes to a single
    #  channel array
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # faces : Video sequence submitted to the prediction (including the subject's face)
    #
    # model : the model trained to make predictions
    #
    #---------------------------------------------------------------------------------
    # 
    # return : frames normalized
    #
    ##--------------------------------------------------------------------------------
    def convert_video_to_table(self, faces ,model):

        imgs = np.zeros(shape=(model.input_shape[1], self.video.cropSize[0], self.video.cropSize[1], 1))

        # channel extraction
        if (self.video.cropSize[2]<3):
            IMAGE_CHANNELS = 1
        else:
            IMAGE_CHANNELS = self.video.cropSize[2]

        # load images (imgs contains the whole video)
        for j in range(0, model.input_shape[1]):

            # normalization
            if (IMAGE_CHANNELS==3):
                temp = faces[j]/255
                temp = temp[:,:,1]      # only the G component is currently used
            else:
                temp = faces[j] / 255

            imgs[j] = np.expand_dims(temp, 2)

        return imgs

    ##--------------------------------------------------------------------------------
    #
    # checkNbFrames(self, faces, model): Checks and corrects the differences 
    # in the number of frames between the model input and the provided frames.
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # faces : Video sequence submitted to the prediction (including the subject's face)
    #
    # model : the model trained to make predictions
    #
    #---------------------------------------------------------------------------------
    # 
    # return : faces processed
    #
    ##--------------------------------------------------------------------------------
    def checkNbFrames(self, faces, model):
        if (model.input_shape[1] > len(faces)):
            faces = self.interpolation(faces,model.input_shape[1])
        return faces
    
    ##--------------------------------------------------------------------------------
    #
    # formating_data_test(self, model, imgs , freq_bpm) : format the video at the model's
    #  entrance standarts (including video mapping)
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # faces : Video sequence submitted to the prediction (including the subject's face)
    #
    # imgs : frames normalised 
    #
    # model : the model trained to make predictions
    #
    # freq_bpm : array containing the set of classes (representing each bpm) known 
    #   by the model
    #
    #---------------------------------------------------------------------------------
    # 
    # return : sum of predictions
    #
    ##--------------------------------------------------------------------------------
    def formating_data_test(self, model, imgs , freq_bpm):
    
        # output - sum of predictions
        predictions = np.zeros(shape=(len(freq_bpm)))
    
        # Displacement on the x axis
        iteration_x = 0
        # Our position at n + 1 on the X axis
        axis_x = model.input_shape[3]
    
        # width of video
        width = self.video.cropSize[1]
        # height of video
        height = self.video.cropSize[0]
    
        # Browse the X axis
        while axis_x < width:
            # Displacement on the y axis
            axis_y = model.input_shape[2]
            # Our position at n + 1 on the Y axis
            iteration_y = 0
            # Browse the Y axis
            while axis_y < height:
            
                # Start position
                x1 = iteration_x * self.x_step
                y1 = iteration_y * self.y_step
            
                # End position
                x2 = x1 + model.input_shape[3]
                y2 = y1 + model.input_shape[2]
            
                # Cutting 
                face_copy = copy(imgs[0:model.input_shape[1],x1:x2,y1:y2,:])
            
                # randomize pixel locations
                for j in range(model.input_shape[1]):
                    temp = copy(face_copy[j,:,:,:])
                    np.random.shuffle(temp)
                    face_copy[j] = temp
            
                # Checks the validity of cutting
                if(np.shape(face_copy)[1] == model.input_shape[3] and np.shape(face_copy)[2] == model.input_shape[2]):
                    # prediction on the cut part
                    xtest = face_copy - np.mean(face_copy)
                    predictions = predictions + self.get_prediction(model,freq_bpm,xtest)
            
                # increments
                axis_y = y2 + model.input_shape[2]
                iteration_y = iteration_y +1
            # increments    
            axis_x = x2 + model.input_shape[3]
            iteration_x = iteration_x + 1
        
        return predictions  

    ##--------------------------------------------------------------------------------
    #
    # get_idx(self, h) : Get the index of the maximum value of a prediction
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # h : Array (here a prediction)
    #
    #---------------------------------------------------------------------------------
    # 
    # return : index of the maximum value of an array
    #
    ##--------------------------------------------------------------------------------
    def get_idx(self, h):
        idx =0
        maxi = -1
        #find label associated
        for i in range(0, len(h)):
            if maxi < h[i]:
                idx = i
                maxi = h[i]
        return idx  

    ##--------------------------------------------------------------------------------
    #
    # get_prediction(self, model, freq_bpm, xtest) :  Making a prediction
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # model : the model trained to make predictions
    #
    # freq_bpm : array containing the set of classes (representing each bpm) known 
    #   by the model
    #
    # xtest : model input
    #
    #---------------------------------------------------------------------------------
    # 
    # return : A prediction 
    #
    ##--------------------------------------------------------------------------------   
    def get_prediction(self, model, freq_bpm, xtest):
        idx = 0
        # model.predict
        input_tensor = tf.convert_to_tensor(np.expand_dims(xtest, 0))
        h = model(input_tensor)
        h = h.numpy() 
        #Binary prediction
        res = np.zeros(shape=(76))
        idx = self.get_idx(h[0])
        res[idx] = 1
        return res     
 
    ##--------------------------------------------------------------------------------
    #
    # interpolation(self, faces, nb_frames_required) :  Adding frames by interpolation
    #
    #---------------------------------------------------------------------------------
    #
    # self : VHRMethod
    #
    # faces : Video sequence submitted to the prediction (including the subject's face)
    #
    # nb_frames_required : Number of missing frames
    #
    #---------------------------------------------------------------------------------
    # 
    # return : A new video sequence with the correct size to be an input of the model
    #
    ##--------------------------------------------------------------------------------
    def interpolation(self, faces, nb_frames_required):
        # find the number of missing images
        diff_frames = nb_frames_required - len(faces)

        # adding images to a random place
        place_interpolation = np.random.randint(1, len(faces), size=(diff_frames))
        for p in place_interpolation:
            faces = np.insert(faces, p, faces[p], axis=0) 

        # memory management
        del place_interpolation

        return faces           
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/02/04

@author: remi carmigniani,kemil Belhadji, C. Prétot
"""
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from moviepy.editor import *
import matplotlib.lines as lines
import pickle
import os
from scipy import linalg  #pour la triangulation
#global paramerters


def hisEqulColor(img):
	ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
	return img


def reject_outliers(data, m=2.):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d / mdev if mdev else 0.
	data_range = np.arange(len(data))
	idx_list = data_range[s >= m]
	return data[s < m], idx_list


class camera3D:
	#camera class with 3D calibration
	def __init__(self,
	             pathVid=None,
	             pathCalib='laser_3D/cameraA1_calib_3D.p',
	             t_zero=0,
	             z_plane=0,
	             xzero=1,
	             yzero=2,
	             scale=300,
	             W=7800,
	             H=1500):
		'''
        Classe camera3D
        load a camera with calibration file and path to the video
        camera3D contains :
        - pathVid : path to the video file
        - pathCalib : path to the calibration file used
        - cap : the video capture
        - t_zero : the synchronization time (useful when several camera are used
        - mtx : intrinseq calibration matrix (focal and optical center)
        - optimal_camera_matrix : intrinseq calibration matrix (focal and optical center) after distortion correction
        - dist : distortion correction vector
        - translation_vector, rotation_vector : extrinseq parameters
        - x0 : ref position for perspective correction (0, 5,10,15,20 m)
        - z0 : 1.5m if aerial and -1.5m if submarine
        - z_plane : plane used for the perspective stitching. default is 0 (center of the swimming lane)
        - M : perspective matrix
        - raw_image : direct image from the camera
        - undistImage : image corrected for the distortion
        - corrImage : image for stitching
        - xzero : origin used (start -1m from the wall)
        - yzero : vertical origin (2m above the surface)
        - scale : 300 pixel =1m
        - W,H : size full stitched image
        '''
		self.pathVid = pathVid
		self.pathCalib = pathCalib
		self.ret = True
		#load video
		if pathVid == None:
			self.cap = None
		else:
			self.cap = cv2.VideoCapture(pathVid)
		#synchronisation time
		self.t_zero = t_zero
		#load calibration and init calib
		calib_result_pickle = pickle.load(open(pathCalib, "rb"))
		#create calibration input
		self.mtx = calib_result_pickle["mtx"]
		self.optimal_camera_matrix = calib_result_pickle[
		    "optimal_camera_matrix"]
		self.dist = calib_result_pickle["dist"]

		real_coord = calib_result_pickle["calib_real"]
		self.rotation_vector = calib_result_pickle["rotation_vector"]
		self.translation_vector = calib_result_pickle["translation_vector"]

		self.x0 = np.round(np.min(real_coord[:, 0]) / 5) * 5
		if np.min(real_coord[:, 1]) < 0:
			self.z0 = -1.5
		else:
			self.z0 = 1.5
		self.z_plane = z_plane

		#proj param
		self.xzero = xzero
		self.yzero = yzero
		self.scale = scale
		self.W = W
		self.H = H
		#compute a first M Matrix
		self.computeM()
		self.computeMTopDown()
		self.computeMSingle()
		self.raw_image = None
		self.undistImage = None
		self.corrImage = None
		#image correction
		self.equalizer = False
		#time
		self.time_s = 0
		self.cam = 0

	def computeMSingle(self):
		'''
        construct the perspective matrix to project the view in -1 meters to 6 meters from x0
        and resolution is 300*7meters
        '''
		dist_coeffs = np.zeros((4, 1))
		x0 = self.x0
		z0 = self.z0
		z_plane = 0  #always centered (it does not matter)
		self.plane_w = np.float32([[x0, 0, z_plane], [x0 + 5, 0, z_plane],
		                           [x0 + 5, z0, z_plane], [x0, z0, z_plane],
		                           [x0, 0, z_plane]]).reshape(-1, 3)
		(self.plane_image,
		 jacobian) = cv2.projectPoints(self.plane_w,
		                               self.rotation_vector,
		                               self.translation_vector,
		                               self.optimal_camera_matrix,
		                               dist_coeffs)
		self.plane_image = self.plane_image.reshape(-1, 2)
		#create src and dst
		src = self.plane_image[:-1]

		dst = (self.plane_w[:-1, :2] * np.float32([1, -1]) +
		       np.float32([self.xzero, self.yzero])) * self.scale
		if self.x0 > 0:
			dst[:, 0] = dst[:, 0] - np.min(
			    dst[:, 0]) + 1.5 * self.xzero * self.scale
			dst[:, 1] = dst[:, 1] - np.min(dst[:, 1]) + 1 * self.scale
		else:
			dst[:, 0] = dst[:, 0] - np.min(
			    dst[:, 0]) + 1 * self.xzero * self.scale
			dst[:, 1] = dst[:, 1] - np.min(dst[:, 1]) + 1 * self.scale
		self.MSingle = cv2.getPerspectiveTransform(src, dst)
		self.MSingleInv = np.linalg.inv(self.MSingle)

	def computeM(self):
		'''
        construct the perspective matrix
        '''
		dist_coeffs = np.zeros((4, 1))
		x0 = self.x0
		z0 = self.z0
		z_plane = self.z_plane
		self.plane_w = np.float32([[x0, 0, z_plane], [x0 + 5, 0, z_plane],
		                           [x0 + 5, z0, z_plane], [x0, z0, z_plane],
		                           [x0, 0, z_plane]]).reshape(-1, 3)
		(self.plane_image,
		 jacobian) = cv2.projectPoints(self.plane_w,
		                               self.rotation_vector,
		                               self.translation_vector,
		                               self.optimal_camera_matrix,
		                               dist_coeffs)
		self.plane_image = self.plane_image.reshape(-1, 2)
		#create src and dst
		src = self.plane_image[:-1]

		dst = (self.plane_w[:-1, :2] * np.float32([1, -1]) +
		       np.float32([self.xzero, self.yzero])) * self.scale

		self.M = cv2.getPerspectiveTransform(src, dst)

	def computeMTopDown(self):
		'''
        construct the perspective matrix for a top down view 
        '''
		dist_coeffs = np.zeros((4, 1))
		x0 = self.x0
		z0 = self.z0
		z_plane = self.z_plane
		self.plane_topdown = np.float32([[x0, 0, -1.25], [x0 + 5, 0, -1.25],
		                                 [x0 + 5, 0, 1.25], [x0, 0, 1.25],
		                                 [x0, 0, -1.25]]).reshape(-1, 3)
		(self.plane_topdown_image,
		 jacobian) = cv2.projectPoints(self.plane_topdown,
		                               self.rotation_vector,
		                               self.translation_vector,
		                               self.optimal_camera_matrix,
		                               dist_coeffs)
		self.plane_topdown_image = self.plane_topdown_image.reshape(-1, 2)
		#create src and dst
		src = self.plane_topdown_image[:-1]

		dst = (self.plane_topdown[:-1, [0, 2]] * np.float32([1, -1]) +
		       np.float32([self.xzero, 1.25])) * self.scale
		self.M_topdown = cv2.getPerspectiveTransform(src, dst)

	def updateZPlane(self, z_plane):
		self.z_plane = z_plane
		self.computeM()

	def undistRawImage(self):
		'''
        distortion correction
        '''
		if self.equalizer:
			self.raw_image = hisEqulColor(self.raw_image)
		self.undistImag = cv2.undistort(self.raw_image,
		                                self.mtx,
		                                self.dist,
		                                None,
		                                self.optimal_camera_matrix)

	def corrDistImage(self):
		'''
        perspective correction for stitching
        '''
		self.corrImage = cv2.warpPerspective(self.undistImag,
		                                     self.M, (self.W, self.H),
		                                     flags=cv2.INTER_LINEAR)

	def corrDistImageSingle(self):
		'''
        perspective correction for single cam
        '''
		self.corrImageSingle = cv2.warpPerspective(
		    self.undistImag,
		    self.MSingle, (self.scale * (8) + self.scale // 2, 4 * self.scale),
		    flags=cv2.INTER_LINEAR)

	def getFrameAtTime(self, time_s):
		'''
        update the time and the raw_video
        '''
		self.time_s = time_s + self.t_zero
		if self.cap == None:
			self.raw_image = np.zeros((10, 10, 3), np.uint8)
		else:
			time_ms = self.time_s * 1000
			#frame_rate = cap.get(cv2.CAP_PROP_FPS)
			#frame_msec = 1000 / frame_rate
			self.cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)  #-frame_msec)

			self.ret, self.raw_image = self.cap.read()
			#corr colors
			#self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
	def getFrameAtTimeFull(self, time_s):
		'''
        do All image corrections
        '''
		self.getFrameAtTime(time_s)
		self.undistRawImage()
		self.corrDistImage()

	def undistAndCorrImage(self):
		'''
        do All image corrections
        '''
		self.undistRawImage()
		self.corrDistImage()

	def getNextRawFrame(self):
		'''
        do All image corrections
        '''
		self.time_s += 0.02
		self.ret, self.raw_image = self.cap.read()

	def getNextFrame(self):
		'''
        do All image corrections
        '''
		self.time_s += 0.02
		self.ret, self.raw_image = self.cap.read()
		#corr colors
		#self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
		if self.ret:
			self.undistRawImage()
			self.corrDistImage()

	def displayRawImage(self,
	                    show_axis=False,
	                    show_plane=False,
	                    show_topdown=False):
		plt.figure(figsize=(10, 10))
		plt.imshow(self.undistImag)

		if show_axis:
			axis = np.float32([[self.x0, 0, 0], [self.x0 + 1, 0, 0],
			                   [self.x0, np.sign(self.z0) * 1,
			                    0], [self.x0, 0, -1]]).reshape(-1, 3)
			(axis_image,
			 jacobian) = cv2.projectPoints(axis,
			                               self.rotation_vector,
			                               self.translation_vector,
			                               self.optimal_camera_matrix,
			                               np.zeros((4, 1)))
			Orig = axis_image[0][0]
			ex = axis_image[1][0]
			ey = axis_image[2][0]
			ez = axis_image[3][0]
			plt.arrow(Orig[0],
			          Orig[1],
			          ex[0] - Orig[0],
			          ex[1] - Orig[1],
			          head_width=40,
			          color='r',
			          length_includes_head=True)
			plt.arrow(Orig[0],
			          Orig[1],
			          ey[0] - Orig[0],
			          ey[1] - Orig[1],
			          head_width=40,
			          color='b',
			          length_includes_head=True)
			plt.arrow(Orig[0],
			          Orig[1],
			          ez[0] - Orig[0],
			          ez[1] - Orig[1],
			          head_width=40,
			          color='purple',
			          length_includes_head=True)
		if show_plane:
			plt.fill(self.plane_image[:, 0],
			         self.plane_image[:, 1],
			         '-r',
			         alpha=0.2)
			plt.plot(self.plane_image[:, 0], self.plane_image[:, 1], '-r')
		if show_topdown:
			plt.fill(self.plane_topdown_image[:, 0],
			         self.plane_topdown_image[:, 1],
			         '-b',
			         alpha=0.2)
			plt.plot(self.plane_topdown_image[:, 0],
			         self.plane_topdown_image[:, 1],
			         '-b')


class multiCam:
	#multi camera for 10 cameras
	def __init__(self,
	             pathRace=None,
	             pathRootCalib='laser_3D/',
	             z_plane=0,
	             xzero=1,
	             yzero=2,
	             scale=300,
	             W=7800,
	             H=1500,
	             selectedCam=np.ones((2, 5))):
		'''
        Classe multiCam
        load all the cameras using the class cam
        multiCam contains :
        - pathRace : path to all the cameras
        - pathRootCalib : path to the calibration folder
        - tabCam : table with each available camera (None if not available)
        - activeCam : the camera active at the current time
        - time_s : the current time 
        - t_zero_tab : the synchronization time (useful when several camera are used
        
        - z_plane : plane used for the perspective stitching. default is 0 (center of the swimming lane)
        - xzero : origin used (start -1m from the wall)
        - yzero : vertical origin (2m above the surface)
        - scale : 300 pixel =1m
        - W,H : size full stitched image
        - stitchImage : stitch images
        - selectedCam : if 1 the cam is activated else not
        '''

		self.pathRace = pathRace
		self.pathRootCalib = pathRootCalib
		self.z_plane = z_plane
		#load t_zeros file:
		if os.path.isfile(pathRace + '/t_zero_cams.txt'):
			#print('\tLoad camera synchronization')
			self.t_zero_tab = np.loadtxt(pathRace + '/t_zero_cams.txt')
			self.not_sync = False
		else:
			print('\tCameras are not synchronized... set to zero')
			self.t_zero_tab = np.zeros(10)
			self.not_sync = True
		#load the cameras
		#A1:
		if os.path.isfile(pathRace + '/camera_A01.mp4'):
			camA1 = camera3D(pathVid=pathRace+'/camera_A01.mp4',pathCalib=pathRootCalib+'/cameraA1_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[0],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camA1 = camera3D(
			    pathVid=None,
			    pathCalib=pathRootCalib + '/cameraA1_calib_3D.p',
			)
		#A2:
		if os.path.isfile(pathRace + '/camera_A02.mp4'):
			camA2 = camera3D(pathVid=pathRace+'/camera_A02.mp4',pathCalib=pathRootCalib+'/cameraA2_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[1],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camA2 = camera3D(
			    pathVid=None,
			    pathCalib=pathRootCalib + '/cameraA1_calib_3D.p',
			)
		#A3:
		if os.path.isfile(pathRace + '/camera_A03.mp4'):
			camA3 = camera3D(pathVid=pathRace+'/camera_A03.mp4',pathCalib=pathRootCalib+'/cameraA3_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[2],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camA3 = camera3D(
			    pathVid=None,
			    pathCalib=pathRootCalib + '/cameraA1_calib_3D.p',
			)
		#A4:
		if os.path.isfile(pathRace + '/camera_A04.mp4'):
			camA4 = camera3D(pathVid=pathRace+'/camera_A04.mp4',pathCalib=pathRootCalib+'/cameraA4_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[3],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camA4 = camera3D(
			    pathVid=None,
			    pathCalib=pathRootCalib + '/cameraA1_calib_3D.p',
			)
		#A5:
		if os.path.isfile(pathRace + '/camera_A05.mp4'):
			camA5 = camera3D(pathVid=pathRace+'/camera_A05.mp4',pathCalib=pathRootCalib+'/cameraA5_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[4],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camA5 = camera3D(
			    pathVid=None,
			    pathCalib=pathRootCalib + '/cameraA1_calib_3D.p',
			)
		#B1:
		if os.path.isfile(pathRace + '/camera_B01.mp4'):
			camB1 = camera3D(pathVid=pathRace+'/camera_B01.mp4',pathCalib=pathRootCalib+'/cameraB1_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[5],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camB1 = camera3D(pathVid=None,
			                 pathCalib=pathRootCalib + '/cameraB1_calib_3D.p')
		#B2:
		if os.path.isfile(pathRace + '/camera_B02.mp4'):
			camB2 = camera3D(pathVid=pathRace+'/camera_B02.mp4',pathCalib=pathRootCalib+'/cameraB2_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[6],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camB2 = camera3D(pathVid=None,
			                 pathCalib=pathRootCalib + '/cameraB2_calib_3D.p')
		#B3:
		if os.path.isfile(pathRace + '/camera_B03.mp4'):
			camB3 = camera3D(pathVid=pathRace+'/camera_B03.mp4',pathCalib=pathRootCalib+'/cameraB3_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[7],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camB3 = camera3D(pathVid=None,
			                 pathCalib=pathRootCalib + '/cameraB3_calib_3D.p')
		#B4:
		if os.path.isfile(pathRace + '/camera_B04.mp4'):
			camB4 = camera3D(pathVid=pathRace+'/camera_B04.mp4',pathCalib=pathRootCalib+'/cameraB4_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[8],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camB4 = camera3D(pathVid=None,
			                 pathCalib=pathRootCalib + '/cameraB2_calib_3D.p')
		#B5:
		if os.path.isfile(pathRace + '/camera_B05.mp4'):
			camB5 = camera3D(pathVid=pathRace+'/camera_B05.mp4',pathCalib=pathRootCalib+'/cameraB5_calib_3D.p',\
                                                    t_zero=self.t_zero_tab[9],z_plane = z_plane,\
                                                    xzero=xzero,yzero =yzero,scale=scale,W=W,H=H)
		else:
			camB5 = camera3D(pathVid=None,
			                 pathCalib=pathRootCalib + '/cameraB2_calib_3D.p')
		self.camAll = np.array([[camA1, camA2, camA3, camA4, camA5],
		                        [camB1, camB2, camB3, camB4, camB5]])
		#proj param
		self.xzero = xzero
		self.yzero = yzero
		self.scale = scale
		self.W = W
		self.H = H
		#time
		self.time_s = 0
		self.selectedCam = np.ones((2, 5))
		self.updateCam()
		self.selectedCam = selectedCam
		#check that setcam is available:
		for i_row in range(2):
			for j_row in range(5):
				if self.selectedCam[i_row, j_row] == 1:
					if self.camAll[i_row, j_row].cap == None:
						self.selectedCam[i_row, j_row] = 0

	def updateCamThread(self):
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].getFrameAtTime(self.time_s)

	def undistAndCorrThread(self):
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].undistAndCorrImage()

	def updateCam(self):
		self.ret = True
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].getFrameAtTimeFull(self.time_s)
					self.ret = self.ret * self.camAll[i_row, j_col].ret

	def updateCamSingle(self):
		self.ret = True
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].getFrameAtTime(self.time_s)
					self.ret = self.ret * self.camAll[i_row, j_col].ret
					if self.camAll[i_row, j_col].ret:
						self.camAll[i_row, j_col].undistRawImage()
						self.camAll[i_row, j_col].corrDistImageSingle()

	def updateCamNext(self):
		self.time_s += 0.02
		self.ret = True
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].getNextFrame()
					self.ret = self.ret * self.camAll[i_row, j_col].ret

	def updateCamNextSingle(self):
		self.time_s += 0.02
		self.ret = True
		for i_row in range(2):
			for j_col in range(5):
				if self.selectedCam[i_row, j_col] == 1:
					self.camAll[i_row, j_col].getNextRawFrame()
					self.ret = self.ret * self.camAll[i_row, j_col].ret
					if self.camAll[i_row, j_col].ret:
						self.camAll[i_row, j_col].undistRawImage()
						self.camAll[i_row, j_col].corrDistImageSingle()

	def updateZPlane(self, z_plane):
		'''
        update the z_plane for all cameras (even if inactive)
        '''
		self.z_plane = z_plane
		for i_row in range(2):
			for j_col in range(5):
				self.camAll[i_row, j_col].updateZPlane(z_plane)

	def createStitchAerial(self):
		i_row = 0
		merged = np.zeros((self.H, self.W, 3), np.uint8)
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				temp = np.copy(self.camAll[i_row, j_col].corrImage)
				if j_col < 4 and self.selectedCam[i_row, j_col + 1] == 1:
					#need to mask for the next image
					temp[:, (self.xzero + (j_col+1) * 5) * self.scale:] = 0
				if j_col > 0 and self.selectedCam[i_row, j_col - 1] == 1:
					#need to mask for the previous image
					temp[:, :(self.xzero + (j_col) * 5) * self.scale] = 0
				merged += temp
		self.stitchA = merged

	def createStitchBottom(self):
		i_row = 1
		merged = np.zeros((self.H, self.W, 3), np.uint8)
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				temp = np.copy(self.camAll[i_row, j_col].corrImage)
				if j_col < 4 and self.selectedCam[i_row, j_col + 1] == 1:
					#need to mask for the next image
					temp[:, (self.xzero + (j_col+1) * 5) * self.scale:] = 0
				if j_col > 0 and self.selectedCam[i_row, j_col - 1] == 1:
					#need to mask for the previous image
					temp[:, :(self.xzero + (j_col) * 5) * self.scale] = 0
				merged += temp
		self.stitchB = merged

	def createStitchBottomFromSingle(self):
		i_row = 1
		merged = np.copy(self.stitchB)
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				if j_col == 0:
					merged[300:, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[:300 + 900, :, :]
				else:
					merged[
					    300:, (j_col*5 + 1) * 300:(j_col*5 + 1 + 6) *
					    300, :] = self.camAll[
					        i_row, j_col].corrImageSingle[:300 + 900, 300:, :]
		self.stitchB = merged

	def createStitchAerialFromSingle(self):
		i_row = 0
		merged = np.copy(self.stitchA)
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				if j_col == 0:
					merged[:1050, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[150:, :, :]
				else:
					merged[:1050,(j_col*5+1)*300:(j_col*5+1+6)*300,:]=self.camAll[i_row,j_col].corrImageSingle[150:,300:,:]
		self.stitchA = merged

	def createStitchFullFromSingle(self):
		merged = np.copy(self.stitchFull)
		self.createStitchAerialFromSingle()
		self.createStitchBottomFromSingle()
		i_row = 0
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				if j_col == 0 and self.selectedCam[1, j_col] == 0:
					merged[:1050, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[150:, :, :]
				elif j_col == 0 and self.selectedCam[1, j_col] == 1:
					merged[:600, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[150:750, :, :]
				elif self.selectedCam[1, j_col] == 0:
					merged[:1050,(j_col*5+1)*300:(j_col*5+1+6)*300,:]=self.camAll[i_row,j_col].corrImageSingle[150:,300:,:]
				elif self.selectedCam[1, j_col] == 1:
					merged[:600,(j_col*5+1)*300:(j_col*5+1+6)*300,:]=self.camAll[i_row,j_col].corrImageSingle[150:750,300:,:]
		i_row = 1
		for j_col in range(5):
			if self.selectedCam[i_row, j_col] == 1:
				#on ajoute l'image
				if j_col == 0 and self.selectedCam[0, j_col] == 0:
					merged[300:, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[:300 + 900, :, :]
				elif j_col == 0 and self.selectedCam[0, j_col] == 1:
					merged[600:, :7 * self.scale, :] = self.camAll[
					    i_row, j_col].corrImageSingle[300:300 + 900, :, :]
				elif self.selectedCam[0, j_col] == 0:
					merged[
					    300:, (j_col*5 + 1) * 300:(j_col*5 + 1 + 6) *
					    300, :] = self.camAll[
					        i_row, j_col].corrImageSingle[:300 + 900, 300:, :]
				elif self.selectedCam[0, j_col] == 1:
					merged[
					    600:, (j_col*5 + 1) * 300:(j_col*5 + 1 + 6) *
					    300, :] = self.camAll[i_row, j_col].corrImageSingle[
					        300:300 + 900, 300:, :]
		self.stitchFull = merged

	def createStitchFull(self):
		self.stitchFull = np.zeros((self.H, self.W, 3), np.uint8)
		self.createStitchAerial()
		self.createStitchBottom()
		self.stitchFull[:self.yzero * self.
		                scale, :] = self.stitchA[:self.yzero * self.scale, :]
		self.stitchFull[self.yzero * self.scale:, :] = self.stitchB[
		    self.yzero * self.scale:, :]

	def create4windowPlayer(self,
	                        cam_selected=[[0, 0], [0, 1], [1, 0], [1, 1]]):
		im = np.zeros((1080, 1920, 3), np.uint8)
		count = 0
		for cam in cam_selected:
			im_cam = self.camAll[cam[0], cam[1]].raw_image
			im_cam = cv2.resize(im_cam, (1920 // 2, 1080 // 2),
			                    interpolation=cv2.INTER_AREA)
			if count == 0:
				im[:540, :960, :] = im_cam
			elif count == 1:
				im[:540, 960:, :] = im_cam
			elif count == 2:
				im[540:, :960, :] = im_cam
			else:
				im[540:, 960:, :] = im_cam
			count += 1
		return im

	def setCam(self, cam=1):
		'''
        Select a set of 4 camera (used for the tracking)
        '''
		self.selectedCam[:, :] = 0  #deselect all cams
		self.cam = cam
		if cam == 1:
			self.selectedCam[:, 0:2] = 1  #select cam A1 B1 A2 B2
		elif cam == 2:
			self.selectedCam[:, 1:3] = 1
		elif cam == 3:
			self.selectedCam[:, 2:4] = 1
		elif cam == 4:
			self.selectedCam[:, 3:] = 1
		else:
			self.selectedCam[:, :] = 1  #select all
			self.cam = 0  #0 all cam
		#check that setcam is available:
		for i_row in range(2):
			for j_row in range(5):
				if self.selectedCam[i_row, j_row] == 1:
					if self.camAll[i_row, j_row].cap == None:
						self.selectedCam[i_row, j_row] = 0

	def getPoseTopDownViewAerial(self,
	                             networktab,
	                             i_cam,
	                             dz_plane=0.05,
	                             th=0.3,
	                             n_min=1):
		'''
        find where is the swimmer roughly in the lane
        convert the data to a top down view to get new_zplane
        dz_plane : resolution selon la position dans la ligne (5cm sur 2.5m!)
        '''
		#get the matrix
		M = self.camAll[0, i_cam].M
		M_topdown = self.camAll[0, i_cam].M_topdown
		#convert the position to pixel in the original aerial view
		#we use the head and hips markers (usually visible from top view)
		#hips right
		i_hips_L = 2
		i_hips_R = 5
		i_head = 0
		flag_prob_head = 3 + 3*i_head + 2
		flag_prob_R = 3 + 3*i_hips_R + 2
		flag_prob_L = 3 + 3*i_hips_L + 2
		I_FLAG = (networktab[:, flag_prob_R] >
		          th) * (networktab[:, flag_prob_L] > th)

		I_FLAG_H = (networktab[:, flag_prob_head] > th)

		flag_xpos = 3 + 3*i_hips_R
		flag_ypos = 3 + 3*i_hips_R + 1
		pos11 = networktab[I_FLAG, flag_xpos:flag_ypos + 1]

		print(networktab[I_FLAG, flag_ypos + 1])
		#hips left
		flag_xpos = 3 + 3*i_hips_L
		flag_ypos = 3 + 3*i_hips_L + 1
		pos8 = networktab[I_FLAG, flag_xpos:flag_ypos + 1]
		pos = np.copy((pos11+pos8) / 2)
		print(networktab[I_FLAG, flag_ypos + 1])
		#midshould

		flag_xpos = 3 + 3*i_head
		flag_ypos = 3 + 3*i_head + 1
		pos0 = networktab[I_FLAG_H, flag_xpos:flag_ypos + 1]
		pos = np.append(pos, pos0, axis=0)
		print(networktab[I_FLAG_H, flag_ypos + 1])

		print(np.nanmean(pos[:, 1]))

		pos_org = np.zeros_like(pos)
		pos_org[:,0],pos_org[:,1],sc=np.linalg.inv(M)@np.array([(25+self.xzero)*self.scale-pos[:,0],pos[:,1],1])
		pos_org[:, 0] = pos_org[:, 0] / sc
		pos_org[:, 1] = pos_org[:, 1] / sc
		#now convert to topdown view to get the z_plane
		pos_topdown = np.zeros_like(pos_org)
		pos_topdown[:,0],pos_topdown[:,1],sc=M_topdown@np.array([pos_org[:,0],pos_org[:,1],1])
		pos_topdown[:, 0] = pos_topdown[:, 0] / sc
		pos_topdown[:, 1] = pos_topdown[:, 1] / sc
		if len(pos_topdown) > n_min:
			print('here')
			#remove outliers:
			print(pos_topdown[:, 1])
			pos_final, _ = reject_outliers(pos_topdown[:, 1], m=3.)
			print(pos_final)
			z_plane_new = np.round(
			    (1.25 - np.nanmedian(pos_final) / self.scale) /
			    dz_plane) * dz_plane
			dz_plane_new = np.round((np.nanstd(pos_final) / self.scale) /
			                        dz_plane * 2) * dz_plane / 2
			if dz_plane_new > 3 * dz_plane:
				z_plane_new = -10
		else:
			z_plane_new = -10
			dz_plane_new = -10
		return z_plane_new, dz_plane_new, pos_topdown

	def triangulateUnderWaterView(self, head_pos_1, head_pos_2, dz_plane=0.05):
		'''
        we triangulate the position of the swimmer in 3D with 2 camera and return the optimal z_plane for the stitch
        '''
		M_1 = self.camAll[1, self.cam - 1].M
		M_2 = self.camAll[1, self.cam].M
		#convert to pos in undistIm
		head_pos_1_org = np.zeros_like(head_pos_1)
		head_pos_1_org[:,0],head_pos_1_org[:,1],sc=np.linalg.inv(M_1)@np.array([(25+self.xzero)*self.scale-head_pos_1[:,0],head_pos_1[:,1],1])
		head_pos_1_org[:, 0] = head_pos_1_org[:, 0] / sc
		head_pos_1_org[:, 1] = head_pos_1_org[:, 1] / sc
		head_pos_2_org = np.zeros_like(head_pos_1)
		head_pos_2_org[:,0],head_pos_2_org[:,1],sc=np.linalg.inv(M_2)@np.array([(25+self.xzero)*self.scale-head_pos_2[:,0],head_pos_2[:,1],1])
		head_pos_2_org[:, 0] = head_pos_2_org[:, 0] / sc
		head_pos_2_org[:, 1] = head_pos_2_org[:, 1] / sc
		#construct the matrix
		R1 = self.camAll[1, self.cam - 1].rotation_vector
		R1 = cv2.Rodrigues(R1)[0]
		T1 = self.camAll[1, self.cam - 1].translation_vector
		mtx1 = self.camAll[1, self.cam - 1].optimal_camera_matrix
		#RT B1.
		RT1 = np.concatenate([R1, T1], axis=-1)
		P1 = mtx1 @ RT1  #projection matrix for B1

		#RT for B2.
		R2 = self.camAll[1, self.cam].rotation_vector
		R2 = cv2.Rodrigues(R2)[0]
		T2 = self.camAll[1, self.cam].translation_vector
		mtx2 = self.camAll[1, self.cam].optimal_camera_matrix
		RT2 = np.concatenate([R2, T2], axis=-1)
		P2 = mtx2 @ RT2  #projection matrix for B2
		pos3D = np.empty((0, 3))
		for i in range(len(head_pos_1)):
			res3D = DLT(P1, P2, head_pos_1_org[i, :], head_pos_2_org[i, :])
			pos3D = np.append(pos3D, [res3D], axis=0)
		z_plane_new = np.round(pos3D[:, 2].mean() / dz_plane) * dz_plane
		return z_plane_new

	def outputNetwork(self, cam=1):
		'''
        return the images used for the network
        '''
		image_i = np.copy(self.stitchFull)
		image_top = np.copy(self.stitchA)
		image_below = np.copy(self.stitchB)
		if cam == 1:
			Araw1 = np.copy(self.camAll[0, 0].corrImage)
			Araw2 = np.copy(self.camAll[0, 1].corrImage)
			Braw1 = np.copy(self.camAll[1, 0].corrImage)
			Braw2 = np.copy(self.camAll[1, 1].corrImage)
		elif cam == 2:
			Araw1 = np.copy(self.camAll[0, 1].corrImage)
			Araw2 = np.copy(self.camAll[0, 2].corrImage)
			Braw1 = np.copy(self.camAll[1, 1].corrImage)
			Braw2 = np.copy(self.camAll[1, 2].corrImage)
		elif cam == 3:
			Araw1 = np.copy(self.camAll[0, 2].corrImage)
			Araw2 = np.copy(self.camAll[0, 3].corrImage)
			Braw1 = np.copy(self.camAll[1, 2].corrImage)
			Braw2 = np.copy(self.camAll[1, 3].corrImage)
		elif cam == 4:
			Araw1 = np.copy(self.camAll[0, 3].corrImage)
			Araw2 = np.copy(self.camAll[0, 4].corrImage)
			Braw1 = np.copy(self.camAll[1, 3].corrImage)
			Braw2 = np.copy(self.camAll[1, 4].corrImage)
		else:
			Araw1 = np.copy(self.camAll[0, 0].corrImage)
			Araw2 = np.copy(self.camAll[0, 1].corrImage)
			Braw1 = np.copy(self.camAll[1, 0].corrImage)
			Braw2 = np.copy(self.camAll[1, 1].corrImage)

		#flip images
		image_i = cv2.flip(image_i, 1)
		image_top = cv2.flip(image_top, 1)
		image_below = cv2.flip(image_below, 1)
		Araw1 = cv2.flip(Araw1, 1)
		Araw2 = cv2.flip(Araw2, 1)
		Braw1 = cv2.flip(Braw1, 1)
		Braw2 = cv2.flip(Braw2, 1)
		return image_i[:1350,:,:],image_top[:1350,:,:],image_below[:1350,:,:],Araw1[:1350,:,:],Araw2[:1350,:,:],Braw1[:1350,:,:],Braw2[:1350,:,:]

		#return image_i[:1350,:,:],image_top[:1350,:,:],image_below[:1350,:,:],Araw1,Araw2,Braw1,Braw2
	def stitchNetwork2cams(self, Araw, Braw, icam=0, z_plane=0):
		if abs(self.z_plane - z_plane) > 0.01:
			self.updateZPlane(z_plane)
		self.camAll[0, icam].raw_image = Araw
		self.camAll[0, icam].undistAndCorrImage()
		self.camAll[1, icam].raw_image = Braw
		self.camAll[1, icam].undistAndCorrImage()
		self.selectedCam[:, :] = 0
		self.selectedCam[:, icam] = 1
		self.createStitchFull()
		image_i = np.copy(self.stitchA)
		image_i = cv2.flip(image_i, 1)
		self.setCam(cam=self.cam)
		return image_i[:1350, :, :]

	def corrNetwork(self, Araw1, Araw2, Braw1, Braw2, cam=1, z_plane=0):
		if abs(self.z_plane - z_plane) > 0.01:
			self.updateZPlane(z_plane)
		if cam == 1:
			self.camAll[0, 0].raw_image = Araw1
			self.camAll[0, 0].undistAndCorrImage()
			self.camAll[0, 1].raw_image = Araw2
			self.camAll[0, 1].undistAndCorrImage()
			self.camAll[1, 0].raw_image = Braw1
			self.camAll[1, 0].undistAndCorrImage()
			self.camAll[1, 1].raw_image = Braw2
			self.camAll[1, 1].undistAndCorrImage()
		elif cam == 2:
			self.camAll[0, 1].raw_image = Araw1
			self.camAll[0, 1].undistAndCorrImage()
			self.camAll[0, 2].raw_image = Araw2
			self.camAll[0, 2].undistAndCorrImage()
			self.camAll[1, 1].raw_image = Braw1
			self.camAll[1, 1].undistAndCorrImage()
			self.camAll[1, 2].raw_image = Braw2
			self.camAll[1, 2].undistAndCorrImage()
		elif cam == 3:
			self.camAll[0, 2].raw_image = Araw1
			self.camAll[0, 2].undistAndCorrImage()
			self.camAll[0, 3].raw_image = Araw2
			self.camAll[0, 3].undistAndCorrImage()
			self.camAll[1, 2].raw_image = Braw1
			self.camAll[1, 2].undistAndCorrImage()
			self.camAll[1, 3].raw_image = Braw2
			self.camAll[1, 3].undistAndCorrImage()
		elif cam == 3:
			self.camAll[0, 3].raw_image = Araw1
			self.camAll[0, 3].undistAndCorrImage()
			self.camAll[0, 4].raw_image = Araw2
			self.camAll[0, 4].undistAndCorrImage()
			self.camAll[1, 3].raw_image = Braw1
			self.camAll[1, 3].undistAndCorrImage()
			self.camAll[1, 4].raw_image = Braw2
			self.camAll[1, 4].undistAndCorrImage()
		self.createStitchFull()
		image_i = np.copy(self.stitchFull)
		image_i = cv2.flip(image_i, 1)
		return image_i[:1350, :, :]


'''
OLD CODE TO DELETE LATER : 
'''


def unwarpchronophoto(img, src, dst, testing, wFinal, hFinal):
	h, w = img.shape[:2]
	# use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
	M = cv2.getPerspectiveTransform(src, dst)
	# use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img,
	                             M, (wFinal, hFinal),
	                             flags=cv2.INTER_LINEAR)

	if testing:
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
		f.subplots_adjust(hspace=.2, wspace=.05)
		ax1.imshow(img)
		x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
		y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
		ax1.plot(x,
		         y,
		         color='red',
		         alpha=0.4,
		         linewidth=3,
		         solid_capstyle='round',
		         zorder=2)
		#ax1.set_ylim([h, 0])
		#ax1.set_xlim([0, w])
		ax1.set_title('Original Image', fontsize=30)
		ax2.imshow(cv2.flip(warped, 1))
		ax2.set_title('Unwarped Image', fontsize=30)
		plt.show()
	else:
		return warped, M


def correctImage(imag,
                 W=7800,
                 H=1500,
                 calibFile='Cameras_Calibration/laser_3D/cameraB5_calib_3D.p',
                 z_plane=0):
	calib_result_pickle = pickle.load(open(calibFile, "rb"))
	mtx = calib_result_pickle["mtx"]
	optimal_camera_matrix = calib_result_pickle["optimal_camera_matrix"]
	dist = calib_result_pickle["dist"]
	undistImag = cv2.undistort(imag, mtx, dist, None, optimal_camera_matrix)
	real_coord = calib_result_pickle["calib_real"]
	rotation_vector = calib_result_pickle["rotation_vector"]
	translation_vector = calib_result_pickle["translation_vector"]
	#construct the M matrix
	x0 = np.round(np.min(real_coord[:, 0]) / 5) * 5
	if np.min(real_coord[:, 1]) < 0:
		z0 = -1.5
	else:
		z0 = 1.5
	dist_coeffs = np.zeros((4, 1))
	plane_w = np.float32([[x0, 0, z_plane], [x0 + 5, 0,
	                                         z_plane], [x0 + 5, z0, z_plane],
	                      [x0, z0, z_plane], [x0, 0, z_plane]]).reshape(-1, 3)
	(plane_image, jacobian) = cv2.projectPoints(plane_w,
	                                            rotation_vector,
	                                            translation_vector,
	                                            optimal_camera_matrix,
	                                            dist_coeffs)
	plane_image = plane_image.reshape(-1, 2)
	#create src and dst
	src = plane_image[:-1]
	xzero = 1
	yzero = 2
	dst = (plane_w[:-1, :2] * np.float32([1, -1]) +
	       np.float32([xzero, yzero])) * 300
	M = cv2.getPerspectiveTransform(src, dst)

	return cv2.warpPerspective(undistImag, M, (W, H), flags=cv2.INTER_LINEAR)


def correctA1(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraA1_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctA2(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraA2_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctA3(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraA3_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctA4(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraA4_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctA5(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraA5_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctB1(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraB1_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctB2(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraB2_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctB3(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraB3_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctB4(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraB4_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)


def correctB5(imag,
              W=7800,
              H=1500,
              calibFile='laser_3D/cameraB5_calib_3D.p',
              z_plane=0):
	return correctImage(imag, W, H, calibFile, z_plane=z_plane)



def createStitchAtTimeFull(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,z_plane=0):
	frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
	frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
	frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
	frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
	frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
	frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
	frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
	frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
	frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
	frameB5 = getFrameAtTime(capB5, time + t_zero_B5)

	rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
	rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
	rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
	rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
	rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
	rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
	rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
	rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
	rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
	rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)

	A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
	A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
	A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
	A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
	A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
	B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
	B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
	B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
	B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
	B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

	A1C[600:, :] = 0
	A1C[:, 1800:] = 0
	A2C[600:, :] = 0
	A2C[:, :1800] = 0
	A2C[:, 3300:] = 0
	A3C[600:, :] = 0
	A3C[:, :3300] = 0
	A3C[:, 4800:] = 0
	A4C[600:, :] = 0
	A4C[:, :4800] = 0
	A4C[:, 6300:] = 0
	A5C[600:, :] = 0
	A5C[:, :6300] = 0
	B1C[:600, :] = 0
	B1C[:, 1800:] = 0
	B2C[:, :1800] = 0
	B2C[:600, :] = 0
	B2C[:, 3300:] = 0
	B3C[:, :3300] = 0
	B3C[:, 4800:] = 0
	B3C[:600, :] = 0
	B4C[:, :4800] = 0
	B4C[:, 6300:] = 0
	B4C[:600, :] = 0
	B5C[:, :6300] = 0
	B5C[:600, :] = 0

	merged = A1C + A2C + A3C + A4C + A5C + B1C + B2C + B3C + B4C + B5C
	return merged, frameA1, frameB1, frameB2, frameB3

def createStitchAtTime(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,z_plane=0):
	frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
	frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
	frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
	frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
	frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
	frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
	frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
	frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
	frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
	frameB5 = getFrameAtTime(capB5, time + t_zero_B5)

	rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
	rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
	rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
	rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
	rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
	rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
	rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
	rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
	rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
	rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)

	A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
	A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
	A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
	A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
	A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
	B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
	B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
	B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
	B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
	B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

	A1C[600:, :] = 0
	A1C[:, 1800:] = 0
	A2C[600:, :] = 0
	A2C[:, :1800] = 0
	A2C[:, 3300:] = 0
	A3C[600:, :] = 0
	A3C[:, :3300] = 0
	A3C[:, 4800:] = 0
	A4C[600:, :] = 0
	A4C[:, :4800] = 0
	A4C[:, 6300:] = 0
	A5C[600:, :] = 0
	A5C[:, :6300] = 0
	B1C[:600, :] = 0
	B1C[:, 1800:] = 0
	B2C[:, :1800] = 0
	B2C[:600, :] = 0
	B2C[:, 3300:] = 0
	B3C[:, :3300] = 0
	B3C[:, 4800:] = 0
	B3C[:600, :] = 0
	B4C[:, :4800] = 0
	B4C[:, 6300:] = 0
	B4C[:600, :] = 0
	B5C[:, :6300] = 0
	B5C[:600, :] = 0

	merged = A1C + A2C + A3C + A4C + A5C + B1C + B2C + B3C + B4C + B5C
	return merged

def createStitchAtTime_topBottom(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           topBottom=0,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,z_plane=0):

	if topBottom == 1:
		frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
		frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
		frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
		frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
		frameA5 = getFrameAtTime(capA5, time + t_zero_A5)

		rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
		rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
		rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
		rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
		rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)

		A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
		A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
		A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
		A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
		A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)

		A1C[:, 1800:] = 0
		A2C[:, :1800] = 0
		A2C[:, 3300:] = 0
		A3C[:, :3300] = 0
		A3C[:, 4800:] = 0
		A4C[:, :4800] = 0
		A4C[:, 6300:] = 0
		A5C[:, :6300] = 0

		merged = A1C + A2C + A3C + A4C + A5C
	elif topBottom == -1:
		frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
		frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
		frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
		frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
		frameB5 = getFrameAtTime(capB5, time + t_zero_B5)

		rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
		rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
		rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
		rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
		rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)

		B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
		B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
		B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
		B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
		B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

		B1C[:, 1800:] = 0
		B2C[:, :1800] = 0
		B2C[:, 3300:] = 0
		B3C[:, :3300] = 0
		B3C[:, 4800:] = 0
		B4C[:, :4800] = 0
		B4C[:, 6300:] = 0
		B5C[:, :6300] = 0

		merged = B1C + B2C + B3C + B4C + B5C
	else:
		frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
		frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
		frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
		frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
		frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
		frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
		frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
		frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
		frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
		frameB5 = getFrameAtTime(capB5, time + t_zero_B5)

		rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
		rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
		rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
		rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
		rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
		rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
		rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
		rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
		rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
		rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)

		A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
		A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
		A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
		A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
		A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
		B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
		B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
		B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
		B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
		B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

		A1C[600:, :] = 0
		A1C[:, 1800:] = 0
		A2C[600:, :] = 0
		A2C[:, :1800] = 0
		A2C[:, 3300:] = 0
		A3C[600:, :] = 0
		A3C[:, :3300] = 0
		A3C[:, 4800:] = 0
		A4C[600:, :] = 0
		A4C[:, :4800] = 0
		A4C[:, 6300:] = 0
		A5C[600:, :] = 0
		A5C[:, :6300] = 0
		B1C[:600, :] = 0
		B1C[:, 1800:] = 0
		B2C[:, :1800] = 0
		B2C[:600, :] = 0
		B2C[:, 3300:] = 0
		B3C[:, :3300] = 0
		B3C[:, 4800:] = 0
		B3C[:600, :] = 0
		B4C[:, :4800] = 0
		B4C[:, 6300:] = 0
		B4C[:600, :] = 0
		B5C[:, :6300] = 0
		B5C[:600, :] = 0

		merged = A1C + A2C + A3C + A4C + A5C + B1C + B2C + B3C + B4C + B5C
	return merged

def createStitchAtTimeTopBelow(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,z_plane=0):
	frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
	frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
	frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
	frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
	frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
	frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
	frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
	frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
	frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
	frameB5 = getFrameAtTime(capB5, time + t_zero_B5)

	rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
	rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
	rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
	rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
	rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
	rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
	rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
	rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
	rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
	rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)

	A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
	A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
	A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
	A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
	A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
	B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
	B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
	B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
	B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
	B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

	#stitch top view
	A1C[:, 1800:] = 0
	A2C[:, :1800] = 0
	A2C[:, 3300:] = 0
	A3C[:, :3300] = 0
	A3C[:, 4800:] = 0
	A4C[:, :4800] = 0
	A4C[:, 6300:] = 0
	A5C[:, :6300] = 0
	mergedTop = np.copy(A1C + A2C + A3C + A4C + A5C)

	#stitch underwater
	B1C[:, 1800:] = 0
	B2C[:, :1800] = 0
	B2C[:, 3300:] = 0
	B3C[:, :3300] = 0
	B3C[:, 4800:] = 0
	B4C[:, :4800] = 0
	B4C[:, 6300:] = 0
	B5C[:, :6300] = 0
	mergedUnder = np.copy(B1C + B2C + B3C + B4C + B5C)

	A1C[600:, :] = 0
	A2C[600:, :] = 0
	A3C[600:, :] = 0
	A4C[600:, :] = 0
	A5C[600:, :] = 0

	B1C[:600, :] = 0
	B2C[:600, :] = 0
	B3C[:600, :] = 0
	B4C[:600, :] = 0
	B5C[:600, :] = 0

	merged = A1C + A2C + A3C + A4C + A5C + B1C + B2C + B3C + B4C + B5C
	return merged, mergedTop, mergedUnder
def createStitchAtTimeTopBelow_Boost(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           cam=1,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,z_plane=0):
	if cam == 1:
		frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
		frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
		frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
		frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
		rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
		rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
		rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
		rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
		A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
		A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
		B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
		B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
		Araw1 = np.copy(A1C)
		Araw2 = np.copy(A2C)
		Braw1 = np.copy(B1C)
		Braw2 = np.copy(B2C)

		A1C[:, 1800:] = 0
		A2C[:, :1800] = 0
		B1C[:, 1800:] = 0
		B2C[:, :1800] = 0
		mergedTop = np.copy(A1C + A2C)
		mergedUnder = np.copy(B1C + B2C)
		A1C[600:, :] = 0
		A2C[600:, :] = 0
		B1C[:600, :] = 0
		B2C[:600, :] = 0
		merged = A1C + A2C + B1C + B2C
	elif cam == 2:
		frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
		frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
		frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
		frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
		rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
		rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
		rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
		rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
		A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
		A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
		B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
		B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
		Araw1 = np.copy(A2C)
		Araw2 = np.copy(A3C)
		Braw1 = np.copy(B2C)
		Braw2 = np.copy(B3C)

		A2C[:, 3300:] = 0
		A3C[:, :3300] = 0
		mergedTop = np.copy(A2C + A3C)
		B2C[:, 3300:] = 0
		B3C[:, :3300] = 0
		mergedUnder = np.copy(B2C + B3C)
		A2C[600:, :] = 0
		A3C[600:, :] = 0
		B2C[:600, :] = 0
		B3C[:600, :] = 0
		merged = A2C + A3C + B2C + B3C
	elif cam == 3:
		frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
		frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
		frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
		frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
		rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
		rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
		rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
		rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
		A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
		A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
		B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
		B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
		Araw1 = np.copy(A3C)
		Araw2 = np.copy(A4C)
		Braw1 = np.copy(B3C)
		Braw2 = np.copy(B4C)
		A3C[:, 4800:] = 0
		A4C[:, :4800] = 0
		mergedTop = np.copy(A3C + A4C)
		B3C[:, 4800:] = 0
		B4C[:, :4800] = 0
		mergedUnder = np.copy(B3C + B4C)
		A3C[600:, :] = 0
		A4C[600:, :] = 0
		B3C[:600, :] = 0
		B4C[:600, :] = 0
		merged = A3C + A4C + B3C + B4C
	elif cam == 4:
		frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
		frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
		frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
		frameB5 = getFrameAtTime(capB5, time + t_zero_B5)
		rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
		rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
		rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
		rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)
		A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
		A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
		B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
		B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)
		Araw1 = np.copy(A4C)
		Araw2 = np.copy(A5C)
		Braw1 = np.copy(B4C)
		Braw2 = np.copy(B5C)
		A4C[:, 6300:] = 0
		A5C[:, :6300] = 0
		mergedTop = np.copy(A4C + A5C)
		B4C[:, 6300:] = 0
		B5C[:, :6300] = 0
		mergedUnder = np.copy(B4C + B5C)
		A4C[600:, :] = 0
		A5C[600:, :] = 0
		B4C[:600, :] = 0
		B5C[:600, :] = 0
		merged = A4C + A5C + B4C + B5C
	else:
		merged,mergedTop,mergedUnder = createStitchAtTimeTopBelow(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                                       time,\
                                       t_zero_A1,t_zero_A2,t_zero_A3,t_zero_A4,t_zero_A5,\
                                       t_zero_B1,t_zero_B2,t_zero_B3,t_zero_B4,t_zero_B5,\
                                       W,H,z_plane=z_plane)
	return merged, mergedTop, mergedUnder, Araw1, Araw2, Braw1, Braw2

def createStitchAtTime_selectCam(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                           time,\
                           cam=1,\
                           topBottom=0,\
                           t_zero_A1=0,t_zero_A2=0,t_zero_A3=0,t_zero_A4=0,t_zero_A5=0,\
                           t_zero_B1=0,t_zero_B2=0,t_zero_B3=0,t_zero_B4=0,t_zero_B5=0,\
                           W=7800,H=1500,
                           z_plane=0):
	if topBottom == 0:
		if cam == 1:
			frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
			frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
			rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
			rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
			A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
			B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
			A1C[600:, :] = 0
			B1C[:600, :] = 0
			merged = A1C + B1C
		elif cam == 2:
			frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
			frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
			rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
			rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
			A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
			B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)

			A2C[600:, :] = 0
			B2C[:600, :] = 0
			merged = A2C + B2C
		elif cam == 3:
			frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
			frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
			rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
			rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
			A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
			B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
			A3C[600:, :] = 0
			B3C[:600, :] = 0
			merged = A3C + B3C
		elif cam == 4:
			frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
			frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
			rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
			rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
			A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
			B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)

			A4C[600:, :] = 0
			B4C[:600, :] = 0
			merged = A4C + B4C
		elif cam == 5:
			frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
			frameB5 = getFrameAtTime(capB5, time + t_zero_B5)
			rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
			rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)
			A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
			B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)

			A5C[600:, :] = 0
			B5C[:600, :] = 0
			merged = A5C + B5C
		else:
			merged = createStitchAtTime_topBottom(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                                                                    time,\
                                                                    topBottom,\
                                                                    t_zero_A1,t_zero_A2,t_zero_A3,t_zero_A4,t_zero_A5,\
                                                                    t_zero_B1,t_zero_B2,t_zero_B3,t_zero_B4,t_zero_B5,\
                                                                    W,H,z_plane=z_plane)
	elif topBottom == 1:  #TOP VIEW ONLY
		if cam == 1:
			frameA1 = getFrameAtTime(capA1, time + t_zero_A1)
			rgbcolorA1 = cv2.cvtColor(frameA1, cv2.COLOR_BGR2RGB)
			A1C = correctA1(rgbcolorA1, W, H, z_plane=z_plane)
			merged = A1C
		elif cam == 2:
			frameA2 = getFrameAtTime(capA2, time + t_zero_A2)
			rgbcolorA2 = cv2.cvtColor(frameA2, cv2.COLOR_BGR2RGB)
			A2C = correctA2(rgbcolorA2, W, H, z_plane=z_plane)
			merged = A2C
		elif cam == 3:
			frameA3 = getFrameAtTime(capA3, time + t_zero_A3)
			rgbcolorA3 = cv2.cvtColor(frameA3, cv2.COLOR_BGR2RGB)
			A3C = correctA3(rgbcolorA3, W, H, z_plane=z_plane)
			merged = A3C
		elif cam == 4:
			frameA4 = getFrameAtTime(capA4, time + t_zero_A4)
			rgbcolorA4 = cv2.cvtColor(frameA4, cv2.COLOR_BGR2RGB)
			A4C = correctA4(rgbcolorA4, W, H, z_plane=z_plane)
			merged = A4C
		elif cam == 5:
			frameA5 = getFrameAtTime(capA5, time + t_zero_A5)
			rgbcolorA5 = cv2.cvtColor(frameA5, cv2.COLOR_BGR2RGB)
			A5C = correctA5(rgbcolorA5, W, H, z_plane=z_plane)
			merged = A5C
		else:
			merged = createStitchAtTime_topBottom(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                                                                    time,\
                                                                    topBottom,\
                                                                    t_zero_A1,t_zero_A2,t_zero_A3,t_zero_A4,t_zero_A5,\
                                                                    t_zero_B1,t_zero_B2,t_zero_B3,t_zero_B4,t_zero_B5,\
                                                                    W,H,z_plane=z_plane)
	elif topBottom == -1:  #TOP VIEW ONLY
		if cam == 1:
			frameB1 = getFrameAtTime(capB1, time + t_zero_B1)
			rgbcolorB1 = cv2.cvtColor(frameB1, cv2.COLOR_BGR2RGB)
			B1C = correctB1(rgbcolorB1, W, H, z_plane=z_plane)
			merged = B1C
		elif cam == 2:
			frameB2 = getFrameAtTime(capB2, time + t_zero_B2)
			rgbcolorB2 = cv2.cvtColor(frameB2, cv2.COLOR_BGR2RGB)
			B2C = correctB2(rgbcolorB2, W, H, z_plane=z_plane)
			merged = B2C
		elif cam == 3:
			frameB3 = getFrameAtTime(capB3, time + t_zero_B3)
			rgbcolorB3 = cv2.cvtColor(frameB3, cv2.COLOR_BGR2RGB)
			B3C = correctB3(rgbcolorB3, W, H, z_plane=z_plane)
			merged = B3C
		elif cam == 4:
			frameB4 = getFrameAtTime(capB4, time + t_zero_B4)
			rgbcolorB4 = cv2.cvtColor(frameB4, cv2.COLOR_BGR2RGB)
			B4C = correctB4(rgbcolorB4, W, H, z_plane=z_plane)
			merged = B4C
		elif cam == 5:
			frameB5 = getFrameAtTime(capB5, time + t_zero_B5)
			rgbcolorB5 = cv2.cvtColor(frameB5, cv2.COLOR_BGR2RGB)
			B5C = correctB5(rgbcolorB5, W, H, z_plane=z_plane)
			merged = B5C
		else:
			merged = createStitchAtTime_topBottom(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                                                                    time,\
                                                                    topBottom,\
                                                                    t_zero_A1,t_zero_A2,t_zero_A3,t_zero_A4,t_zero_A5,\
                                                                    t_zero_B1,t_zero_B2,t_zero_B3,t_zero_B4,t_zero_B5,\
                                                                    W,H,z_plane=z_plane)
	else:
		merged = createStitchAtTime_topBottom(capA1,capA2,capA3,capA4,capA5,capB1,capB2,capB3,capB4,capB5,\
                                                          time,\
                                                          0,\
                                                          t_zero_A1,t_zero_A2,t_zero_A3,t_zero_A4,t_zero_A5,\
                                                          t_zero_B1,t_zero_B2,t_zero_B3,t_zero_B4,t_zero_B5,\
                                                          W,H,z_plane=z_plane)
	return merged


def getFrameAtTime(cap, time_s):
	if cap == None:
		frame = np.zeros((10, 10, 3), np.uint8)
	else:
		time_ms = time_s * 1000
		#frame_rate = cap.get(cv2.CAP_PROP_FPS)
		#frame_msec = 1000 / frame_rate
		cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)  #-frame_msec)

		ret, frame = cap.read()

	return frame


def DLT(P1, P2, point1, point2):
	'''
    inspired from : https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    '''
	A = [
	    point1[1] * P1[2, :] - P1[1, :],
	    P1[0, :] - point1[0] * P1[2, :],
	    point2[1] * P2[2, :] - P2[1, :],
	    P2[0, :] - point2[0] * P2[2, :]
	]
	A = np.array(A).reshape((4, 4))
	#print('A: ')
	#print(A)

	B = A.transpose() @ A
	U, s, Vh = linalg.svd(B, full_matrices=False)

	return Vh[3, 0:3] / Vh[3, 3]

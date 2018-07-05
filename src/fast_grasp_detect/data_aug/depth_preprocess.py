import numpy as np
import IPython
import cv2

COUNT = 0
def depth_to_3ch(img):

	w,h = img.shape
	new_img = np.zeros([w,h,3])


	img = img.flatten()
	img[img>1000] = 0 

	img = img.reshape([w,h])
	for i in range(3):
		new_img[:,:,i] = img

	return new_img


def depth_scaled_to_255(img):
	
	img = 255.0/np.max(img)*img
	img = np.array(img,dtype=np.uint8)



	for i in range(3):
		img[:,:,i] = cv2.equalizeHist(img[:,:,i])

	cv2.imshow('debug.png',img)
	cv2.waitKey(30)
	
	return img


def depth_to_net_dim(img):

	img = depth_to_3ch(img)
	img = depth_scaled_to_255(img)

	return img

def datum_to_net_dim(datum):

	datum['d_img'] = depth_to_net_dim(datum['d_img'])
	return datum



if __name__ == '__main__':

   data = np.load("data_36_1.npy")
   data = data.all()

   img = depth_to_net_dim(data['d_img'])


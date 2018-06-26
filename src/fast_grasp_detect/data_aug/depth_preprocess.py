import numpy as np
import IPython



def depth_to_3ch(img):
	new_img = np.zeros([img.shape[0],img.shape[1],3])

	for i in range(3):
		new_img[:,:,i] = img

	return new_img


def depth_scaled_to_255(img):
	img = 255.0/np.max(img)*img
	return img


def depth_to_net_dim(img):

	img = depth_to_3ch(img)
	img = depth_scaled_to_255(img)

	return img

def datum_to_net_dim(datum):

	datum['d_img'] = depth_to_net_dim(datum['d_img'])



if __name__ == '__main__':

   data = np.load("data_36_1.npy")
   data = data.all()

   img = depth_to_net_dim(data['d_img'])
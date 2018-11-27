from tqdm import tqdm
import cv2
import numpy as np 
import keras.backend as K
import matplotlib.pyplot as plt

def generate_grad_cam(img_tensor, model, class_index, activation_layer):
	inp = model.input
	y_c = model.output.op.inputs[0][0, class_index]
	A_k = model.get_layer(activation_layer).output
	get_output = K.function([inp], [A_k, K.gradients(y_c, A_k)[0], model.output])

	## 이미지 텐서를 입력해서
	## 해당 액티베이션 레이어의 아웃풋(a_k)과
	## 소프트맥스 함수 인풋의 a_k에 대한 gradient를 구한다.
	[conv_output, grad_val, model_output] = get_output([img_tensor])

	## 배치 사이즈가 1이므로 배치 차원을 없앤다.
	conv_output = conv_output[0]
	grad_val = grad_val[0]

	## 구한 gradient를 픽셀 가로세로로 평균내서 a^c_k를 구한다.
	weights = np.mean(grad_val, axis=(0, 1))

	## 추출한 conv_output에 weight를 곱하고 합하여 grad_cam을 얻는다.
	grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
	for k, w in enumerate(weights):
		grad_cam += w * conv_output[:, :, k]

	grad_cam = cv2.resize(grad_cam, (224, 224))

	## ReLU를 씌워 음수를 0으로 만든다.
	grad_cam = np.maximum(grad_cam, 0)

	## image를 원래값으로 바꿔줌
	image = img_tensor[0,:]
	image = np.uint8(image)

	## Grad CAM을 원래값으로 바꿔줌
	grad_cam = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*grad_cam),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)

	## Blend image
	grad_cam = cv2.addWeighted(grad_cam,0.5,image,0.5,0)

	return grad_cam


def create_grad_cam(model, layer_indices, class_index, imgs, test_idx):
	# set init value
	grad_imgs = []
	layer_names = layer_indices
	test_imgs = imgs
	test_idx = test_idx

	# test_idx 만큼 레이어 별 Grad CAM 생성
	for i in tqdm(test_idx):
		grad_i_imgs = []
		for layer_name in layer_names:
			img_tensor = test_imgs[i].reshape(1,224,224,3)
			grad_img = generate_grad_cam(img_tensor, model, class_index, layer_name)
			grad_i_imgs.append(grad_img)

		grad_imgs.append(grad_i_imgs)

	return grad_imgs


def grad_cam_plot(origin_imgs, grad_cams, layer_names, pred, labels, test_idx, filenames):
	
	col = len(layer_names) + 1
	row = len(test_idx)
	f, axarr = plt.subplots(row,col)
	for i in range(row):
		for j in range(col):
			# 첫번째 열은 오리지날 이미지 출력 
			if j%col == 0:
				sub_plt = axarr[i,int(j%col)]
				sub_plt.axis('off')
				img = cv2.cvtColor(origin_imgs[test_idx[i]],cv2.COLOR_GRAY2RGB)
				sub_plt.imshow(img)
				sub_plt.set_title(filenames[test_idx[i]])
			else:
				R = 'moyamoya' if labels[test_idx[i]] else 'control'
				P = 'moyamoya' if pred[test_idx[i]]  else 'control'
				main_title = 'R: {0:} P: {1:} \nProb: {2:.2f}'.format(R,P,prob*100)
				sub_plt = axarr[i,int(j%col)]
				sub_plt.axis('off')
				sub_plt.imshow(grad_cams[i][j-1])
				sub_plt.set_title(main_title)

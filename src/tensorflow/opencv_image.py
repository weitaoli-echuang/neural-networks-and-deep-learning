# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np

crop_region = [0, 1001, 0, 1001]

clip_region = {
	"start_row": 1,
	"end_row": 1,
	"start_col": 1,
	"end_col": 1
}

learn_region = []


def list_events():
	events = [i for i in dir(cv2) if 'EVENT' in i]
	print(events)


def draw_circle(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(param[0], (x, y), 100, (255, 0, 0), -1)


def test_draw_circle():
	img = np.zeros((512, 512, 3), np.uint8)
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', draw_circle, [img])

	while (1):
		cv2.imshow('image', img)
		if cv2.waitKey(20) & 0xFF == 27:
			break
	cv2.destroyAllWindows()


def on_mouse(event, x, y, flags, param):
	global clip_region
	org = param[0]
	if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
		print("row %d, col %d", y, x)
		clip_region['start_row'] = y
		clip_region['start_col'] = x
		cv2.circle(org, (x, y), 10, (255, 0, 0))
	elif event == cv2.EVENT_LBUTTONUP and (flags & cv2.EVENT_FLAG_SHIFTKEY):
		print("row %d, col %d", y, x)
		clip_region['end_row'] = y
		clip_region['end_col'] = x
		start_pos = (clip_region['start_col'], clip_region['start_row'])
		end_pos = (clip_region['end_col'], clip_region['end_row'])
		if start_pos != end_pos:
			cv2.rectangle(org, end_pos, start_pos, (0, 0, 255), 2)
	elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_CTRLKEY):
		print("row %d, col %d", y, x)
		learn_region.append((x, y))
		cv2.circle(org, (x, y), 10, (0, 255, 0))
	pass


def main():
	file_name = 'image/image000835small.png'

	image = cv2.imread(file_name, 1)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', image)

	x = tf.Variable(image, name='x')
	model = tf.global_variables_initializer()

	with tf.Session() as sess:
		x = tf.transpose(x, perm=[1, 0, 2])
		sess.run(model)
		result = sess.run(x)

	cv2.namedWindow('result', cv2.WINDOW_NORMAL)
	cv2.imshow('result', result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	img = cv2.imread('image/image000835smallsmall.png')
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.setMouseCallback('image', on_mouse, [img])

	while 1:
		cv2.imshow('image', img)
		if cv2.waitKey(20) & 0xFF == 27:
			break

	# crop_img = img[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3], :]
	# cv2.namedWindow('croped_image', cv2.WINDOW_NORMAL)
	# cv2.imshow('croped_image', crop_img)

	cv2.destroyAllWindows()

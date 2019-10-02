import cv2
import numpy as np


def make_coordinates(image, line_parameters): 
	# print(line_parameters)
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3.0/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	if lines is not None:
		for line in lines: 
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1,x2), (y1,y2), 1)  # fit first-degree polynomial 
			slope = parameters[0]
			interecept = parameters[1]
			if slope < -0.1:
				left_fit.append((slope,interecept))
			elif slope > 0.1:
				right_fit.append((slope,interecept))
	else:
		return None
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	if (np.isnan(left_fit_average).all()) or (np.isnan(right_fit_average).all()):
		print ("Error, found None")
		return None
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

# Edge detection 
def canny(image):
	# Reducing from 3 RGB channel to 1 channel, point is, it is more efficient to process one channel than three
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# Reduce image noise, may mess up edge detection, smoothes the image by getting an average of sorrouding pixels 
	blur = cv2.GaussianBlur(gray, (5,5),0)
	# looks for high contrast and detects edges, in our case lanes. Performs derivitve in x and y direction 
	# cv2.Canny(image, low_threshold, high_threshold)
	canny = cv2.Canny(blur, 50, 150)
	return canny

# Get the regtion of interest to reduce processing region to run faster 
# 400,630  640,375   1000,630   
# old : 		[(600,1260), (1270,675), (2080,1275)]
def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
		[(200,height), (1100,height), (550,250)]
		])
	mask = np.zeros_like(image) # creates an array of zeros with same dimensions as the original image
	cv2.fillPoly(mask, polygons, 255)
	# do a bitwise AND operation e.g.  [0,0,1,0] AND [0,0,0,0] = [1,1,0,1]
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image


def display_lines(image, lines): 
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines: 
			x1 , y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2,y2), (255,0,0), 10)
	return line_image


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
	_, frame = cap.read()
	canny_image = canny(frame)
	croppped_image = region_of_interest(canny_image)
	# smaller bins create better vote  find
	lines = cv2.HoughLinesP(croppped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	averaged_lines = average_slope_intercept(frame, lines)
	if averaged_lines is None:
		continue
	line_image = display_lines(frame, averaged_lines)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # multiple all elements in lane image by 0.8 making the image a bit darker
	cv2.imshow("result", combo_image)
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()




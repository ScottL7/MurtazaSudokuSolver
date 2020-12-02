import cv2
import numpy as np
from tensorflow.keras.models import load_model

DEBUG = False
PREDICTION_THRESHOLD = 0.6  # Original threshold was at 80%

digitDim = 28


# Read the model weights
def initializePredictionModel(model):
    """

    :param model: 1 = Murtaza, 2 = Scott
    :type model: Int
    :return:
    :rtype:
    """
    global digitDim
    if model == 1:
        print("--> Using Murtaza's CNN model")
        digitDim = 28
        return load_model('Resources/myModel.h5')
    else:
        print("--> Using Scott's CNN model")
        digitDim = 32
        return load_model(".", custom_objects=None, compile=True)


# 1 - Preprocessing of the Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Add Gaussian blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # Apply adaptive threshold
    return imgThreshold


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def showStackedImage(imgArray, scale=1):
    cv2.imshow('Sudoku Solution!', stackImages(scale, imgArray))


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50 and area > max_area:  # SRL: Added area > max_area, no need to calc if area < max_area
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints


def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def getPrediction(boxes, model):
    result = []
    for image in boxes:
        # Prepare image for predicted classification
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (digitDim, digitDim))
        img = img / 255  # Normalized image to values between 0 - 1
        img = img.reshape(1, digitDim, digitDim, 1)

        # Get prediction
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probability = np.amax(predictions)
        if DEBUG:
            print(f'Predicted digit: {classIndex}, Probability: {probability}')

        # Save result
        if probability > PREDICTION_THRESHOLD:
            result.append(classIndex[0])
        else:
            result.append(0)

    return result


def displayNumbers(img, nums, color):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(9):
        for y in range(9):
            if nums[(y * 9) + x] != 0:
                cv2.putText(img, str(nums[(y * 9) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


def drawGrid(img):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for i in range(9):
        pt1 = (0, secH*i)
        pt2 = (img.shape[1], secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img

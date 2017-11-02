import numpy as np
import cv2


def calGradient(input, operator):
    shape = np.shape(input)
    rows = shape[0]
    cols = shape[1]
    g = np.zeros((rows, cols))
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            g[i, j] = np.multiply(input[i - 1:i + 2, j - 1:j + 2], operator).sum()
    g = np.asmatrix(g)
    return g

def normalize(magnitude):
    maxValue = magnitude.max()
    minValue = magnitude.min()
    print('max:', maxValue)
    print('min:', minValue)
    shape = np.shape(magnitude)
    rows = shape[0]
    cols = shape[1]
    n = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            n[i, j] = ((magnitude[i, j] - minValue) / (maxValue - minValue)) * 255
    return n

def otsu(input):
    print("Running Otsu's method to get the threshold.")
    input = np.array(input, dtype=np.uint8)
    hist = cv2.calcHist([input], [0], None, [256], [0, 256])
    histNorm = hist.ravel() / hist.max()
    Q = histNorm.cumsum()
    bins = np.arange(256)

    fMin = np.inf
    thresh = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(histNorm, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        if q1 != 0 and q2 != 0:
            b1, b2 = np.hsplit(bins, [i])

            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

            fn = v1 * q1 + v2 * q2
            if fn < fMin:
                fMin = fn
                thresh = i
    print('Threshold:', thresh)
    return thresh


def threshold(input, thresh):
    print('before thresholding', input)

    shape = np.shape(input)
    rows = shape[0]
    cols = shape[1]
    t = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if input[i, j] < thresh:
                t[i, j] = 0
            else:
                t[i, j] = 255
    return t

def houghLine(input):
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = input.shape
    print('shape:', input.shape)
    diagLen = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-diagLen, diagLen, 2 * diagLen)

    cosT = np.cos(thetas)
    sinT = np.sin(thetas)
    numThetas = len(thetas)

    accumulator = np.zeros((int(diagLen) * 2, numThetas), dtype=np.uint64)
    yIdx, xIdx = np.nonzero(input)

    for i in range(len(xIdx)):
        x = xIdx[i]
        y = yIdx[i]
        for tIdx in range(numThetas):
            rho = int(x * cosT[tIdx] + y * sinT[tIdx] + diagLen)
            accumulator[rho, tIdx] += 1

    return diagLen, accumulator, thetas, rhos

def filterLines(accumulator):
    straightLines = []
    for rhoIdx in range(accumulator.shape[0]):
        for thetaIdx in range(accumulator.shape[1]):
            if accumulator[rhoIdx, thetaIdx] > 0.3 * accumMax:
                rho = rhos[rhoIdx]
                theta = thetas[thetaIdx]
                straightLines.append([rho, np.rad2deg(theta), accumulator[rhoIdx, thetaIdx]])
    return straightLines

# If two lines rho and theta's difference are both lower than 5% of the range, lines will be merged.
def mergeLines(diagLen, stLines):
    mergedResult = []
    for i in range(len(stLines)):
        curCloseIdx = [i]
        for j in range(len(stLines)):
            if abs(stLines[i][0] - stLines[j][0]) / (diagLen * 2) < 0.05 and abs(stLines[i][1] - stLines[j][1]) / 90 < 0.05:
                curCloseIdx.append(j)
        if len(curCloseIdx) > 1:
            maxNumIdx = curCloseIdx[0]
            for idx in curCloseIdx:
                if stLines[idx][2] >= stLines[maxNumIdx][2]:
                    maxNumIdx = idx
            if maxNumIdx == curCloseIdx[0]:
                mergedResult.append(stLines[i])
        else:
            mergedResult.append(stLines[i])
    return mergedResult

# Let y = kx + b. [k, b]
def transferToXyCoord(diagLen, stLines):
    stLinesInXyCoord = []
    for stLine in stLines:
        k = -(1/np.tan(np.deg2rad(stLine[1])))
        b = (stLine[0]) / np.sin(np.deg2rad(stLine[1]))
        stLinesInXyCoord.append([k, b])
    return stLinesInXyCoord

def findParallelLinesGroup(stLines):
    parallelLinesGroup = []
    for i in range(len(stLines)):
        for j in range(i + 1, len(stLines)):
            if abs(np.rad2deg(np.arctan(stLines[j][0])) - np.rad2deg(np.arctan(stLines[i][0]))) <= 5:
                parallelLinesGroup.append([stLines[i], stLines[j]])
    return parallelLinesGroup

# Assume two lines: y = k1 * x + b1, y = k2 * x + b2
# Based on calculation, the intersection point is ((b2 - b1) / (k1 - k2), (k2 * b1 - k1 * b2) / (k2 - k1))
def findRectCoords(parallelLinesGroup):
    intersections = []
    for i in range(len(parallelLinesGroup)):
        for j in range(i + 1, len(parallelLinesGroup)):
            intersection = []
            # calculate 4 intersection points
            for idxX in range(2):
                for idxY in range(2):
                    pt = calIntersection(parallelLinesGroup[i][idxX], parallelLinesGroup[j][idxY])
                    intersection.append(pt)
            intersections.append(intersection)
    return intersections


def calIntersection(group1, group2):
    k1 = group1[0]
    b1 = group1[1]
    k2 = group2[0]
    b2 = group2[1]
    # To fix the edge index, we need to add 1 for row index and column index.
    x = int(round((b2 - b1) / (k1 - k2)) + 1)
    y = int(round((k2 * b1 - k1 * b2) / (k2 - k1)) + 1)
    return [x, y]

def findValidIntersection(intersections, input):
    validIntersections = []
    for intersection in intersections:
        validPointCount = 0
        for point in intersection:
            if input[point[1], point[0]] > 0:
                validPointCount += 1
        if validPointCount == 4:
            validIntersections.append(intersection)
        print("Valid point counts:", validPointCount)
    return validIntersections

def drawLines(intersections, imgName):
    input = cv2.imread(imgName, cv2.IMREAD_COLOR)
    for intersection in intersections:
        connectLineIdx = []
        connectLineDistance = []
        for i in range(3):
            for j in range(i + 1, 4):
                connectLineIdx.append([i, j])
                currentDistance = np.sqrt((intersection[j][0] - intersection[i][0]) ** 2 + (intersection[j][1] - intersection[i][1]) ** 2)
                print("Current distance:", currentDistance)
                connectLineDistance.append(currentDistance)
        argSort = np.argsort(np.array(connectLineDistance))
        print("Idx:", connectLineIdx)
        print("Line distance:", connectLineDistance)
        print("arg sort:", argSort)
        for i in range(4):
            lineIdx1 = connectLineIdx[argSort[i]][0]
            lineIdx2 = connectLineIdx[argSort[i]][1]
            print("current line index:", lineIdx1, lineIdx2)
            cv2.line(input, (intersection[lineIdx1][0], intersection[lineIdx1][1]), (intersection[lineIdx2][0], intersection[lineIdx2][1]), (255, 0, 0), 2)
    cv2.imshow('image', input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('TestImage3.jpg', 0)
shape = np.shape(img)
rows = shape[0]
cols = shape[1]
print(rows, cols)
imgn = np.fromfile('TestImage1c.raw', dtype=np.uint8, count=rows*cols)
imgn = np.asmatrix(imgn)
grayImg = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        pos = cols * i + j
        # print(pos, i, j)
        grayImg[i, j] = imgn[0, pos]
# grayImg = np.matrix(grayImg)
print('max:', img.max())
print('min:', img.min())
print('grayscale shape:', grayImg.shape)
print(img)


xOperator = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
yOperator = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gx = calGradient(img, xOperator)
gy = calGradient(img, yOperator)
print(gx.max(), gx.min())
print(gy.max(), gy.min())
magnitude = np.sqrt(np.multiply(gx, gx) + np.multiply(gy, gy))
normalizedMagnitude = normalize(magnitude)
print("normalized max:", normalizedMagnitude.max())
print("normalized min:", normalizedMagnitude.min())
th = otsu(normalizedMagnitude)
t = threshold(normalizedMagnitude, 26)
# th, t2 = cv2.threshold(np.array(normalizedMagnitude, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print('Th matrix:', t2)
print('Threshold pack,', th)

# cv2.imshow('image', t)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(t.shape)
print(t)

diagLen, accumulator, thetas, rhos = houghLine(t)
# print('rhos length:', len(rhos))
print('accumulator shape:', accumulator.shape)
print('max:', accumulator.max(), 'min:', accumulator.min())
accumMax = accumulator.max()
straightLines = filterLines(accumulator)
mergedLines = mergeLines(diagLen, straightLines)
print("straight lines:", straightLines)
print("merged straight lines:", mergedLines)
stLinesInXyCoord = transferToXyCoord(diagLen, mergedLines)
print("Straight lines in xy-coordinate:", stLinesInXyCoord)
parallelLinesGroup = findParallelLinesGroup(stLinesInXyCoord)
print("Parallel lines group:", parallelLinesGroup)
intersections = findRectCoords(parallelLinesGroup)
print("Intersections:", intersections)
validIntersections = findValidIntersection(intersections, img)
print("Valid intersections:", validIntersections)
drawLines(validIntersections, 'TestImage3.jpg')




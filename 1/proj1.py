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
            if accumulator[rhoIdx, thetaIdx] > 0.2 * accumMax:
                rho = rhos[rhoIdx]
                theta = thetas[thetaIdx]
                straightLines.append([rho, np.rad2deg(theta), accumulator[rhoIdx, thetaIdx]])
    return straightLines

# If two lines rho and theta's difference are both lower than 5% of the range, lines will be merged.
def mergeLines(diagLen, stLines):
    mergedResult = []
    calculatedIdx = set()
    l = len(stLines)
    for i in range(l - 1):
        curCloseIdx = [i]
        if i not in calculatedIdx:
            for j in range(i + 1, l):
                if abs(stLines[i][0] - stLines[j][0]) / (diagLen * 2) < 0.05 and abs(
                                stLines[i][1] - stLines[j][1]) / 90 < 0.05:
                    curCloseIdx.append(j)
            if len(curCloseIdx) > 1:
                votesSum = 0
                weightedPhoSum = 0
                weightedThetaSum = 0
                for idx in curCloseIdx:
                    votesSum += stLines[idx][2]
                for idx in curCloseIdx:
                    weightedPhoSum += stLines[idx][0] * stLines[idx][2] / votesSum
                    weightedThetaSum += stLines[idx][1] * stLines[idx][2] / votesSum
                choosedLine = [weightedPhoSum, weightedThetaSum, votesSum / len(curCloseIdx)]
                mergedResult.append(choosedLine)
                for idx in curCloseIdx:
                    calculatedIdx.add(idx)
    for i in range(l):
        if i not in calculatedIdx:
            mergedResult.append(stLines[i])
    return mergedResult

def plotMergedLines(mergedLines, input):
    for line in mergedLines:
        rho = line[0]
        theta = np.deg2rad(line[1])
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(input, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('image', input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # Let y = kx + b. [k, b]
# def transferToXyCoord(diagLen, stLines):
#     stLinesInXyCoord = []
#     for stLine in stLines:
#         k = -(1/np.tan(np.deg2rad(stLine[1])))
#         b = (stLine[0]) / np.sin(np.deg2rad(stLine[1]))
#         stLinesInXyCoord.append([k, b])
#     return stLinesInXyCoord

def findParallelLinesPair(stLines):
    parallelLinesPairs = []
    for i in range(len(stLines)):
        for j in range(i + 1, len(stLines)):
            if abs(stLines[i][1] - stLines[j][1]) <= 3 and abs(int(stLines[i][2]) - int(stLines[j][2])) < 0.15 * (int(stLines[i][2]) + int(stLines[j][2])):
                parallelLinesPairs.append([stLines[i], stLines[j]])
    return parallelLinesPairs

def matchParallelLinesPairs(parallelLinesPairs):
    validPairs = []
    l = len(parallelLinesPairs)
    for i in range(l - 1):
        for j in range(i + 1, l):
            beta_i = (parallelLinesPairs[i][0][1] + parallelLinesPairs[i][1][1]) / 2
            beta_j = (parallelLinesPairs[j][0][1] + parallelLinesPairs[j][1][1]) / 2
            avg_votes_i = (parallelLinesPairs[i][0][2] + parallelLinesPairs[i][1][2]) / 2
            avg_votes_j = (parallelLinesPairs[j][0][2] + parallelLinesPairs[j][1][2]) / 2
            ang_diff = abs(beta_i - beta_j)
            rho_diff_i = abs(parallelLinesPairs[i][0][0] - parallelLinesPairs[i][1][0])
            rho_diff_j = abs(parallelLinesPairs[j][0][0] - parallelLinesPairs[j][1][0])
            if rho_diff_i != 0 and rho_diff_j != 0:
                value1 = abs((rho_diff_i - avg_votes_j * np.sin(np.deg2rad(ang_diff))) / rho_diff_i)
                value2 = abs((rho_diff_j - avg_votes_i * np.sin(np.deg2rad(ang_diff))) / rho_diff_j)
                print("compare:", [value1, value2])
                if max([value1, value2]) < 0.3:
                    validPairs.append([parallelLinesPairs[i], parallelLinesPairs[j]])
    return validPairs


# Assume two lines: rho1 = x cos θ1 + y sin θ1, rho2 = x cos θ2 + y sin θ2
# that is AX = b, where
# A = [cos θ1  sin θ1]   b = |rho1|   X = |x|
#     [cos θ2  sin θ2]       |rho2|       |y|
# Solve x for intersection
def findRectCoords(validPairs, shape):
    intersections = []
    for pair in validPairs:
        sameValue = False
        p1, sameValue = calIntersection(pair[0][0], pair[1][0])
        p2, sameValue = calIntersection(pair[0][0], pair[1][1])
        p3, sameValue = calIntersection(pair[0][1], pair[1][0])
        p4, sameValue = calIntersection(pair[0][1], pair[1][1])
        if not sameValue:
            intersection = [p1, p2, p3, p4]
            intersectionValid = True
            for point in intersection:
                x = point[1]
                y = point[0]
                if x < 0 or x > shape[1] - 1 or y < 0 or y > shape[0] - 1:
                    intersectionValid = False
                    break
            if intersectionValid:
                intersections.append(intersection)
    return intersections

def calIntersection(group1, group2):
    rho1 = group1[0]
    theta1 = group1[1]
    rho2 = group2[0]
    theta2 = group2[1]
    # if rho1 == rho2 and theta1 == theta2:
    #     return None, True
    # print("cos(theta1):", np.cos(np.deg2rad(theta1)))
    A = np.array([[np.cos(np.deg2rad(theta1)), np.sin(np.deg2rad(theta1))], [np.cos(np.deg2rad(theta2)), np.sin(np.deg2rad(theta2))]])
    b = np.array([rho1, rho2])
    try:
        result = np.linalg.solve(A, b)
    except:
        return [0, 0], True
    # To fix the edge index, we need to add 1 for row index and column index.
    x = int(result[0] + 1)
    y = int(result[1] + 1)
    return [x, y], False

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
    num = 0
    for intersection in intersections:
        xSum = 0
        ySum = 0
        for point in intersection:
            xSum += point[1]
            ySum += point[0]
        center = [ySum / 4, xSum / 4]
        angle = []
        for point in intersection:
            dx = point[1] - center[1]
            dy = point[0] - center[0]
            angle.append(np.arctan2(dy, dx))
        plotOrder = np.argsort(np.array(angle))
        print("Angle:", angle)
        print("Plot order:", plotOrder)
        for idx in range(3):
            cv2.line(input, (intersection[plotOrder[idx]][0], intersection[plotOrder[idx]][1]),(intersection[plotOrder[idx + 1]][0], intersection[plotOrder[idx + 1]][1]), (255, 0, 0), 2)
        cv2.line(input, (intersection[plotOrder[3]][0], intersection[plotOrder[3]][1]),(intersection[plotOrder[0]][0], intersection[plotOrder[0]][1]), (255, 0, 0), 2)

        # Plot points and text
        for point in intersection:
            cv2.circle(input, (point[0], point[1]), 2, (0, 0, 255), -11)
            cv2.putText(input, str(num), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        num += 1
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
t = threshold(normalizedMagnitude, 25)
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
print("straight lines:", len(straightLines))
print("merged straight lines:", mergedLines)
# plotMergedLines(mergedLines, img)
parallelLinesGroup = findParallelLinesPair(mergedLines)
print("Parallel lines group:", parallelLinesGroup)
validPairs = matchParallelLinesPairs(parallelLinesGroup)
intersections = findRectCoords(validPairs, shape)
print("Intersections:", intersections)
# validIntersections = findValidIntersection(intersections, img)
print("Valid intersections:", intersections)
drawLines(intersections, 'TestImage3.jpg')




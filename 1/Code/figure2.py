import numpy as np
import cv2

# Detect edges using Sobel operator
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

def non_Maxima_Suppression(m, gx, gy):
    shape = np.shape(m)
    theta = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            theta[i, j] = np.rad2deg(np.arctan2(gy[i, j], gx[i, j]))
    print("Theta:", theta, "max:", theta.max(), "min:", theta.min())
    sector = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            t = theta[i, j]
            if -22.5 < t <= 22.5 or 157.5 < t <= 180 or -180 <= t <= -157.5:
                sector[i, j] = 0
            elif 22.5 < t <= 67.5 or -157.5 < t <= -112.5:
                sector[i, j] = 1
            elif 67.5 < t <= 112.5 or -112.5 < t <= -67.5:
                sector[i, j] = 2
            elif 112.5 < t <= 157.5 or -67.5 < t <= -22.5:
                sector[i, j] = 3
    result = np.zeros((shape[0], shape[1]))
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            if sector[i, j] == 0:
                n1 = m[i, j - 1]
                n2 = m[i, j + 1]
            elif sector[i, j] == 1:
                n1 = m[i + 1, j - 1]
                n2 = m[i - 1, j + 1]
            elif sector[i, j] == 2:
                n1 = m[i - 1, j]
                n2 = m[i + 1, j]
            elif sector[i, j] == 3:
                n1 = m[i - 1, j - 1]
                n2 = m[i + 1, j + 1]
            if m[i, j] > n1 and m[i, j] > n2:
                result[i, j] = m[i, j]
            else:
                result[i, j] = 0
    print("Non-Maxima Suppression", result)
    return result
# Normalize the value of pixels
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

# Use Otsu's method to calculate threshold. Just for consideration.
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

# Generate binary edge map
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

# Do Hough Transform.
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

# Filter out lines whose votes are too low
def filterLines(accumulator, votesThreshold):
    straightLines = []
    accumMax = accumulator.max()
    for rhoIdx in range(accumulator.shape[0]):
        for thetaIdx in range(accumulator.shape[1]):
            if accumulator[rhoIdx, thetaIdx] > votesThreshold * accumMax:
                rho = rhos[rhoIdx]
                theta = thetas[thetaIdx]
                straightLines.append([rho, np.rad2deg(theta), accumulator[rhoIdx, thetaIdx]])
    return straightLines

# Merge close lines. If two lines rho and theta's difference are both lower than 5% of the range, lines will be merged.
def mergeLines(diagLen, stLines, phoDifference, thetaDifference):
    mergedResult = []
    calculatedIdx = set()
    l = len(stLines)
    for i in range(l - 1):
        curCloseIdx = [i]
        if i not in calculatedIdx:
            for j in range(i + 1, l):
                if abs(stLines[i][0] - stLines[j][0]) / (diagLen * 2) < phoDifference and abs(
                                stLines[i][1] - stLines[j][1]) < thetaDifference:
                    curCloseIdx.append(j)
            if len(curCloseIdx) > 1:
                maxNumIdx = curCloseIdx[0]
                closeVotes = 0
                for idx in curCloseIdx:
                    closeVotes += stLines[idx][2]
                    if stLines[idx][2] >= stLines[maxNumIdx][2]:
                        maxNumIdx = idx
                choosedLine = stLines[maxNumIdx]
                choosedLine[2] = closeVotes / len(curCloseIdx)
                mergedResult.append(choosedLine)
                for idx in curCloseIdx:
                    calculatedIdx.add(idx)
    for i in range(l):
        if i not in calculatedIdx:
            mergedResult.append(stLines[i])
    return mergedResult

# Plot merged lines. Just for testing.
def plotSeparateLines(mergedLines, input):
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

# Find parallel lines and pair them.
def findParallelLinesPair(stLines, thetaDifference, threshold):
    parallelLinesPairs = []
    for i in range(len(stLines)):
        for j in range(i + 1, len(stLines)):
            if abs(stLines[i][1] - stLines[j][1]) <= thetaDifference and abs(int(stLines[i][2]) - int(stLines[j][2])) < threshold * (int(stLines[i][2]) + int(stLines[j][2])) / 2:
                parallelLinesPairs.append([stLines[i], stLines[j]])
    return parallelLinesPairs

# Match the parallel line pair.
def matchParallelLinesPairs(parallelLinesPairs, matchThreshold):
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
                if max([value1, value2]) <= matchThreshold:
                    validPairs.append([parallelLinesPairs[i], parallelLinesPairs[j]])
            # validPairs.append([parallelLinesPairs[i], parallelLinesPairs[j]])
    return validPairs

# Find intersections for parallel line pairs.
# Assume two lines: rho1 = x cos θ1 + y sin θ1, rho2 = x cos θ2 + y sin θ2
# that is AX = b, where
# A = [cos θ1  sin θ1]   b = |rho1|   X = |x|
#     [cos θ2  sin θ2]       |rho2|       |y|
# Solve x for intersection
def findIntersectionsOfParallelograms(validPairs, shape):
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
                x = point[0]
                y = point[1]
                if x < 0 or x > shape[1] - 1 or y < 0 or y > shape[0] - 1:
                    intersectionValid = False
                    break
            if intersectionValid:
                intersections.append(intersection)
    return intersections

# Calculate intersections of two lines.
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
        print("Error: Singular")
        return [0, 0], True
    # To fix the edge index, we need to add 1 for row index and column index.
    x = int(result[0] + 2)
    y = int(result[1] + 2)
    return [x, y], False

# Sort intersections based on angles for obtaining sides of parallelograms.
def sortIntersectionsByAngles(intersections):
    sortedIntersections = []
    for intersection in intersections:
        xSum = 0
        ySum = 0
        sortedIntersection = []
        for point in intersection:
            xSum += point[1]
            ySum += point[0]
        center = [ySum / 4, xSum / 4]
        angle = []
        for point in intersection:
            dx = point[1] - center[1]
            dy = point[0] - center[0]
            angle.append(np.arctan2(dy, dx))
        order = np.argsort(np.array(angle))
        for i in range(4):
            sortedIntersection.append(intersection[order[i]])
        sortedIntersections.append(sortedIntersection)
    return sortedIntersections

# Filter out parallelograms whose sides are two short.
def getValidMinDistanceOfIntersections(intersections, shape):
    validDistanceIntersections = []
    for intersection in intersections:
        distances = []
        for i in range(3):
            distances.append(calDistance(intersection[i], intersection[i + 1]))
        distances.append(calDistance(intersection[3], intersection[0]))
        # print("Distances:", distances)
        if min(distances) < shape[0] / 20:
            continue
        validDistanceIntersections.append(intersection)
    return validDistanceIntersections


def calDistance(point1, point2):
    return np.sqrt(abs(point1[0] - point2[0]) ** 2 + abs(point1[1] - point2[1]) ** 2)

# Filter out intersections that could not form a parallelogram
def findValidIntersectionForParallelogram(intersections, input, validPercentageThreshold):
    validIntersections = []
    for intersection in intersections:
        validLinesCount = 0
        for i in range(3):
            # print("intersection selected:", intersection[i])
            validLinesCount += findValidIntersectionForParallelogramHelper(intersection[i], intersection[i + 1],
                                                                           input, validPercentageThreshold)
        validLinesCount += findValidIntersectionForParallelogramHelper(intersection[3], intersection[0], input, validPercentageThreshold)
        if validLinesCount == 4:
            validIntersections.append(intersection)
        print("Valid lines count:", validLinesCount)
    return validIntersections

def findValidIntersectionForParallelogramHelper(point1, point2, input, validPercentageThreshold):
    x1 = point1[0] - 1
    y1 = point1[1] - 1
    x2 = point2[0] - 1
    y2 = point2[1] - 1
    # print("Current points:", [x1, y1], [x2, y2])
    l = max(abs(x1 - x2), abs(y1 - y2))
    x = np.linspace(x1, x2, l)
    y = np.linspace(y1, y2, l)
    totalCount = l
    validCount = 0
    for i in range(totalCount):
        if pointValidAround(input, int(y[i]), int(x[i]), 3):
            validCount += 1
    if totalCount == 0:
        return 0
    validPercentage = validCount / totalCount
    print("Valid Count, Total Count, valid percentage", validCount, totalCount, validPercentage)
    if validPercentage > validPercentageThreshold:
        return 1
    else:
        return 0

# check the n x n square around the point on edge figure, if any pixel > 0, return true
def pointValidAround(input, y, x, n):
    upperLeft = [y - 1, x - 1]
    valid = False
    for i in range(n):
        for j in range(n):
            if input[upperLeft[0] + i, upperLeft[1] + j] > 0:
                valid = True
                break
    return valid

# Merge close parallelograms
def mergeCloseParallelograms(intersections, minimumDistance):
    l = len(intersections)
    mergedIntersections = set()
    result = []
    for i in range(l - 1):
        if i not in mergedIntersections:
            closeIndexes = [i]
            for j in range(i + 1, l):
                closePointsCount = 0
                for pointPos in range(4):
                    if pointPos == 3:
                        if np.sqrt((intersections[i][3][0] - intersections[j][3][0]) ** 2 + (intersections[i][3][1] - intersections[j][3][1]) ** 2) < minimumDistance:
                            closePointsCount += 1
                    else:
                        if np.sqrt((intersections[i][pointPos][0] - intersections[j][pointPos][0]) ** 2 + (intersections[i][pointPos + 1][1] - intersections[j][pointPos + 1][1]) ** 2) < minimumDistance:
                            closePointsCount += 1
                if closePointsCount == 4:
                    closeIndexes.append(j)
            if len(closeIndexes) == 1:
                result.append(intersections[closeIndexes[0]])
            else:
                avgIntersection = []
                for pointPos in range(4):
                    xSum = 0
                    ySum = 0
                    for idx in closeIndexes:
                        xSum += intersections[idx][pointPos][0]
                        ySum += intersections[idx][pointPos][1]
                    avgX = int(xSum / len(closeIndexes))
                    avgY = int(ySum / len(closeIndexes))
                    avgIntersection.append([avgX, avgY])
                result.append(avgIntersection)
            print("close indexed:", closeIndexes)
            for idx in closeIndexes:
                mergedIntersections.add(idx)
    return result


# Plot parallelograms
def plotLines(intersections, imgName):
    input = cv2.imread(imgName, cv2.IMREAD_COLOR)
    num = 0
    for intersection in intersections:
        for idx in range(3):
            cv2.line(input, (intersection[idx][0], intersection[idx][1]), (intersection[idx + 1][0], intersection[idx + 1][1]), (255, 0, 0), 2)
        cv2.line(input, (intersection[3][0], intersection[3][1]), (intersection[0][0], intersection[0][1]), (255, 0, 0), 2)

        # Plot points and text
        for point in intersection:
            cv2.circle(input, (point[0], point[1]), 2, (0, 0, 255), -11)
            cordStr = "(" + str(point[0]) + "," + str(point[1]) + ")"
            cv2.putText(input, cordStr, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))
        num += 1
    cv2.imshow('image', input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

fileName = 'TestImage2c.jpg'
img = cv2.imread(fileName, 0)
shape = np.shape(img)
rows = shape[0]
cols = shape[1]
print(rows, cols)
# imgn = np.fromfile('TestImage2c.raw', dtype=np.uint8, count=rows*cols)
# imgn = np.asmatrix(imgn)
# grayImg = np.zeros((rows, cols))
# for i in range(rows):
#     for j in range(cols):
#         pos = cols * i + j
#         # print(pos, i, j)
#         grayImg[i, j] = imgn[0, pos]
# # grayImg = np.matrix(grayImg)
# print('grayscale shape:', grayImg.shape)
print(img)


xOperator = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
yOperator = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
gx = calGradient(img, xOperator)
gy = calGradient(img, yOperator)
print(gx.max(), gx.min())
print(gy.max(), gy.min())
magnitude = np.sqrt(np.multiply(gx, gx) + np.multiply(gy, gy))
thined = non_Maxima_Suppression(magnitude, gx, gy)
normalizedMagnitude = normalize(thined)
print("normalized max:", normalizedMagnitude.max())
print("normalized min:", normalizedMagnitude.min())
th = otsu(normalizedMagnitude)
t = threshold(normalizedMagnitude, 22)
print('Threshold pack,', th)

# cv2.imshow('image', t)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print(t)
#
diagLen, accumulator, thetas, rhos = houghLine(t)
print('accumulator shape:', accumulator.shape)
print('max:', accumulator.max(), 'min:', accumulator.min())
straightLines = filterLines(accumulator, 0.18)
mergedLines = mergeLines(diagLen, straightLines, 0.03, 3)
print("straight lines:", len(straightLines))
print("merged straight lines:", mergedLines)
# plotSeparateLines(straightLines, img)
# plotSeparateLines(mergedLines, img)
parallelLinesGroup = findParallelLinesPair(mergedLines, 5, 0.8)
print("Parallel lines group:", parallelLinesGroup)
validPairs = matchParallelLinesPairs(parallelLinesGroup, 0.8)
print("Valid pairs:", validPairs)
intersections = findIntersectionsOfParallelograms(validPairs, shape)
print("Intersections:", intersections)
print("Valid intersections:", len(intersections), intersections)
sortedIntersections = sortIntersectionsByAngles(intersections)
validMinDistanceOfIntersections = getValidMinDistanceOfIntersections(sortedIntersections, shape)
validIntersectionsForParallelogram = findValidIntersectionForParallelogram(validMinDistanceOfIntersections, t, 0.8)
mergedIntersectionsForParallelogram = mergeCloseParallelograms(validIntersectionsForParallelogram, 10)
print("Valid Intersections for parallelogram:", len(mergedIntersectionsForParallelogram), mergedIntersectionsForParallelogram)
plotLines(mergedIntersectionsForParallelogram, fileName)



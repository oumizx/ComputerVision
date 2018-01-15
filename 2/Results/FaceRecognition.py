import numpy as np
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Load image examples into matrix.
def loadExamples(input_names):
    R = []
    for input_name in input_names:
        img = cv2.imread(input_name, 0).reshape((1, -1))[0].tolist()
        R.append(img)
    R = np.array(R)
    return R

# Subtract means from images.
def subtract(mean, R):
    A = R - mean.transpose()
    return A

# Calculate eigenfaces of images.
def calculateEigen(A):
    L = np.dot(A, np.transpose(A))
    print(L)
    eigen_values, eigen_vectors = np.linalg.eig(L)
    print("eval:", eigen_values)
    print("evec:", eigen_vectors)
    eigenfaces = np.dot(eigen_vectors, A)
    print("eigen faces:", eigenfaces)
    return eigenfaces

# Transfer images to eigenfaces.
def transferToEigenfaces(A, eigenfaces):
    return np.dot(A, np.transpose(eigenfaces))

# Classify the images.
def recognize(A, U, omega_i):
    omega_I = np.dot(A, np.transpose(U))
    print("PCA coefficients for test images:")
    printPCACoefficients(testing_examples_names, N, omega_I)
    print("omega_I:", omega_I)
    reconstructed_images = np.dot(omega_I, U)
    normalized_reconstructed_images = normalize(reconstructed_images)
    pltMultipleImg(normalized_reconstructed_images, N, h, w, 6)  # plot reconstructed images
    print("reconstructed_images:", reconstructed_images)
    distance_para_img_reconstruction = reconstructed_images - A
    print("distance reconstruction:", distance_para_img_reconstruction)
    print("distance parameters:", distance_para_img_reconstruction.shape)
    d0 = np.sqrt(np.sum(np.square(distance_para_img_reconstruction), axis=1))
    print("d0:", d0)
    di = []
    for cur_omega_i in omega_i:
        cur_distance_para_face_space = omega_I - cur_omega_i
        cur_distance_face_space = np.sqrt(np.sum(np.square(cur_distance_para_face_space), axis=1)).tolist()
        di.append(cur_distance_face_space)
    di = np.array(di).transpose()
    dj = np.amin(di, axis=1)  # dj = min{di}
    di = np.array(di)
    printDistanceDi(M, di)
    predict_sort_number = np.argsort(di, axis=1)[:, 0]
    true_sort_number = np.array([0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 5, -1, 6, 6, 6, 7, -1])
    print("predict sort number:", predict_sort_number)
    print("true sort number:", true_sort_number)
    print("dj:", dj)
    printClassificationResult(training_examples_names, testing_examples_names, d0, dj, predict_sort_number)

# Plot single image.
def pltImg(img):
    img = np.array(img, dtype=np.uint8)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')
    # plt.show()

# Plot multiple images.
def pltMultipleImg(img, nplt, h, w, plot_row_nums):
    for i in range(nplt):
        plt.subplot(plot_row_nums, nplt / plot_row_nums, i + 1)
        pltImg(img[i, :].reshape([h, w]))
    plt.show()

# Normalize the images to range (0, 255).
def normalize(img):
    image = img
    minval = image.min()
    maxval = image.max()
    newimg = (img - minval) * (255.0/(maxval-minval))
    return newimg

# Print PCA coefficients.
def printPCACoefficients(examples_names, example_nums, omega_i):
    for i in range(example_nums):
        print(examples_names[i] + ":" + str(omega_i[i, :]))

# Print distance between input face and training images in the face space di.
def printDistanceDi(M, di):
    for i in range(M):
        print("i = " + str(i) + ", di =" + str(di[i, :]))

# Print classification result.
def printClassificationResult(training_examples_names, testing_examples_names, d0, dj, predict_sort_number):
    N = len(testing_examples_names)
    print("Classification result:")
    print("image to predict: classification result(number), classification result(original image name)")
    for i in range(N):
        if d0[i] > 7e+12:
            result = 'non-face'
        elif dj[i] > 1e+8:
            result = 'unknown face'
        else:
            result = str(predict_sort_number[i]) + ', ' + str(training_examples_names[predict_sort_number[i]])
        print(testing_examples_names[i] + ": " + result)


training_examples_names = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg',
                           'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg',
                           'subject14.normal.jpg', 'subject15.normal.jpg']
testing_examples_names = ['subject01.centerlight.jpg', 'subject01.happy.jpg', 'subject01.normal.jpg',
                          'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.centerlight.jpg',
                          'subject07.happy.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg',
                          'subject11.centerlight.jpg', 'subject11.happy.jpg', 'subject11.normal.jpg',
                          'subject12.normal.jpg', 'subject14.happy.jpg', 'subject14.normal.jpg', 'subject14.sad.jpg',
                          'subject15.normal.jpg', 'apple1_gray.jpg']

M = 8  # number of training examples
N = 18  # number of testing examples
h = 231  # picture height pixels
w = 195  # picture width pixels
R_training = loadExamples(training_examples_names) # training examples' pixel grayscales
m = np.mean(R_training, axis=0)  # mean value of training examples
print("shape of mean:", m.shape)
pltImg(m.reshape([h, w]))  # print mean face image
A_training = subtract(m, R_training)  # training face subtracted from the mean face m
print("R training:", R_training)
print("A training:", A_training)
U = calculateEigen(A_training)  # eigenfaces
normalized_U = normalize(U)
pltMultipleImg(normalized_U, M, h, w, 2)  # plot eigenfaces
omega_i = transferToEigenfaces(A_training, U)  # training examples in eigenfaces
print("print training examples PCA coefficients")
printPCACoefficients(training_examples_names, M, omega_i)
R_testing = loadExamples(testing_examples_names)
A_testing = subtract(m, R_testing)
normalized_A_testing = normalize(A_testing)
pltMultipleImg(normalized_A_testing, N, h, w, 6)  # plot image after subtracting the mean face
recognize(A_testing, U, omega_i)
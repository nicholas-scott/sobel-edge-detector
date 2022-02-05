from PIL import Image, ImageOps
import numpy

sobelXKernel = ""
sobelYKernel = ""
file = "img0.jpg"
HORIZONTAL = 0
DIAGONALRIGHTUP = 1
VERTICAL = 2
DIAGNOLRIGHTDOWN =3

#TODO: Add readme, choosing which image to open 
class Kernel():
    def __init__(self, value):
        self.size = 1
        self.array2D = [[value]] 
        self.k = 0
        self.sum = 1

    def __init__(self):
        self.array2D = None

    def get(self, x, y):
        return self.array2D[x + self.k][y + self.k]

    def set(self, x, y, value):
        self.array2D[x + self.k][y + self.k] = value

    def innitGausian(self, sigma):
        self.size = int((2 * numpy.ceil(3 * sigma)) + 1)
        self.array2D = [[0] * self.size for i in range(self.size)]
        self.k = (self.size - 1) // 2  
        self.sum = 0

        for i in range(-self.k, self.k + 1):
            for j in range(-self.k, self.k + 1):
                distribution = self.calcGausian(i, j, sigma)
                self.sum += distribution
                self.set(i, j, distribution)

    def innitSobelX(self):
        self.size = 3
        self.array2D = [[1, 0 , -1], [2, 0, -2], [1, 0, -1]] 
        self.k = 1
        self.sum = 1
    
    def innitSobelY(self):
        self.size = 3
        self.array2D = [[1, 2 , 1], [0, 0, 0], [-1, -2, -1]] 
        self.k = 1
        self.sum = 1


    def calcGausian(self, x, y, sigma):
        ## TODO: only recalculate the exponent 
        muiltiplier = 1 / (2 * numpy.pi * sigma ** 2)
        exponent = (- (x ** 2 + y ** 2)/(2 * sigma ** 2))
        distribution = muiltiplier * numpy.e ** exponent

        return distribution

def main() :
    colorImg = Image.open(file)
    greyImg = ImageOps.grayscale(colorImg)
    
    myEdgeFilter(greyImg, 1) 
    colorImg.close()

## Takes a PIL image, and does the magic on it given sigma 
def myEdgeFilter(img0, sigma):
    size = img0.size
    data = numpy.asarray(img0)
    
    kernel = Kernel()
    
    ##Apply gausian blurr to greyscale image.  
    kernel.innitGausian(sigma)
    dataGausianFiltered = convolve(data, kernel)

    ## Get x and y sobel gradients of smoothed image 
    kernel.innitSobelX()
    dataGradientX = convolve(dataGausianFiltered, kernel)
    gradientXImage = Image.fromarray(dataGradientX, mode="L")
    gradientXImage.save(f"gradient_x_{file}")

    kernel.innitSobelY()
    dataGradientY = convolve(dataGausianFiltered, kernel)
    gradientYImage = Image.fromarray(dataGradientY, mode="L")
    gradientYImage.save(f"gradient_y_{file}")

    ## Calculate magnitude of gradient at each point
    dataMagnitude = sobelMagnitude(dataGradientX, dataGradientY)
    dataMagnitudeImage = Image.fromarray(dataMagnitude, mode="L")
    dataMagnitudeImage.save(f"gradient_magnitude_{file}")

    #Calculate direction of magnitude, does not create a greyscale image. 
    dataMagnitudeDirections = sobelDirection(dataGradientX, dataGradientY)
    print(dataMagnitudeDirections)

    #Calculate the threshholds using the magnitude and direction
    dataEdgesDetected = getEdges(dataMagnitude, dataMagnitudeDirections)
    print(dataEdgesDetected)
    dataEdgeDetectedImage = Image.fromarray(dataEdgesDetected, mode="L")
    dataEdgeDetectedImage.save(f"edges_{file}")

    img1 = dataEdgeDetectedImage
    return img1

## Takes an imgArr representing the image, convolves it with the kernel. Returns a new imgArr with convolved values. 
def convolve(imgArr, kernel):
    height, width = imgArr.shape
    ##2D array of type 8 bit int 
    convolvedImg = numpy.array([[0] * width for i in range(height)]).astype(numpy.uint8)

    ## Iterate through every pixel in image, apply kernel to that pixel
    for i in range(height):
        for j in range(width):
            convolution = applyKernel(imgArr, kernel, i, j)
            convolvedImg[i][j] = convolution.astype(numpy.uint8)
    return convolvedImg

def applyKernel(imgArr, kernel, i, j):
    convolutionSum = 0
    for u in range(- kernel.k, kernel.k + 1):
        for v in range(- kernel.k, kernel.k + 1) :
            ## Negative out of bounds check
            if(i - u >= 0 and j - v >= 0):
                try:
                    imgData = imgArr[i - u][j - v]
                ## Positive out of bounds catch
                except IndexError:
                    imgData = 0
                convolutionSum += imgData * kernel.get(u, v)
    return convolutionSum / kernel.sum

## Takes two sobel gradeints ands gets the magnitude of the gradient. 
def sobelMagnitude(gradientX, gradientY):
    height, width = gradientX.shape
    dataMagnitudeGradient = numpy.array([[0] * width for i in range(height)]).astype(numpy.uint8)
    for i in range(height):
        for j in range(width):
            dataMagnitudeGradient[i][j] = numpy.sqrt((gradientX[i][j] **2) +  (gradientY[i][j] ** 2))
    return dataMagnitudeGradient

## Creates an array of the directions of the gradient. 
def sobelDirection(gradientX, gradientY):
    height, width = gradientX.shape
    dataSobelDirection = numpy.array([[0] * width for i in range(height)]).astype(numpy.uint8)
    for i in range(height):
        for j in range(width):
            dataSobelDirection[i][j] = sobelDirectionToRegion(numpy.arctan2(gradientY[i][j], gradientX[i][j]))
    return dataSobelDirection

## Gets the direcetion of the neighbours
def sobelDirectionToRegion(direction):
    direction = numpy.degrees(direction) % 180
    if(direction < 22.5):
        return HORIZONTAL
    if(direction < 67.5):
        return DIAGONALRIGHTUP
    if(direction < 112.5):
        return VERTICAL
    if(direction < 157.5):
        return DIAGNOLRIGHTDOWN
    else:
        return HORIZONTAL

## Returns all edge pixels using non-maxima supression 
def getEdges(gradientMagnitude, gradientDirections):
    height, width = gradientMagnitude.shape
    dataSobelEdges = numpy.array([[0] * width for i in range(height)]).astype(numpy.uint8)
    for i in range(height):
        for j in range(width):
            neighbour1, neighbour2 = getThreshHoldNeighbours(i, j, gradientDirections[i][j])
            isThreshHold = checkThreshHold(gradientMagnitude, (i, j), neighbour1, neighbour2)
            if(not isThreshHold):
                dataSobelEdges[i][j] = 0
            else:
                dataSobelEdges[i][j] = gradientMagnitude[i][j]
    return dataSobelEdges

## Returns the coordinates of the theshhold neighbours 
def getThreshHoldNeighbours(i, j, gradientDirection):
    if(gradientDirection == HORIZONTAL):
        ## Return the left and right neighbour 
        return (i, j + 1), (i, j - 1) 
    if(gradientDirection == DIAGONALRIGHTUP):
        ## Return Diagnol bottle left, top right
        return (i + 1, j - 1), (i - 1, j + 1)   
    if(gradientDirection == VERTICAL):
        ## Return top and bottom neighbour
        return (i + 1, j), (i - 1, j)  
    else:
        ## Return top left and bottom right
        return (i - 1, j - 1), (i + 1, j + 1) 

## Checks if the threshold is met 
def checkThreshHold(gradientMagnitude, center, neighbour1, neighbour2):
    centerMag = gradientMagnitude[center[0], center[1]]
    for neighbour in [neighbour1, neighbour2]:
        x, y = neighbour
        if(x >= 0 and y >= 0):
            try:
                if(gradientMagnitude[x][y] > centerMag):
                    ## Is not an edge value
                    return False
            ## Positive out of bounds catch, no gradient for out of bounds neighbour
            except IndexError:
                continue
    #Is an edge value
    return True
     ## Negative out of bounds check
    

if __name__ == "__main__":
    main() 
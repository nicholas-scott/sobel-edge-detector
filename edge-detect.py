from PIL import Image, ImageOps
from numpy import arctan2, array, asarray, ceil, degrees, e, exp, gradient, pi, power, sqrt
import numpy

file = "cat2.jpg"
threshhold = 30
threshhold2 = 30
sigma = 1

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
        self.size = int((2 * ceil(3 * sigma)) + 1)
        self.array2D = [[0] * self.size for i in range(self.size)]
        self.k = (self.size - 1) // 2  
        self.sum = 1

        for i in range(-self.k, self.k + 1):
            for j in range(-self.k, self.k + 1):
                distribution = self.calcGausian(i, j, sigma)
                #self.sum += distribution
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
    
    def innitAverage(self, size):
        self.size = size
        self.array2D = [[1] * self.size for i in range(self.size)] 
        self.k = (self.size - 1) // 2 
        self.sum = 1

    def innitValues(self, size, array2D, k, sum):
        self.size = size
        self.array2D = array2D 
        self.k = k
        self.sum = sum

    def innitStickFilter(self, i):
        options = {
            0: lambda : self.innitValues(5, [[0, 0 ,0 , 0, 0], [0, 0 ,0 , 0, 0], [1/5, 1/5 ,1/5 , 1/5, 1/5], [0, 0 ,0 , 0, 0], [0, 0 ,0 , 0, 0]], 2, 1),
            1: lambda : self.innitValues(5, [[0, 0 ,0 , 0, 0], [0, 0 ,0 , 1/5, 1/5], [0, 0 ,1/5 , 0, 0], [1/5, 1/5 ,0 , 0, 0], [0, 0 ,0 , 0, 0]], 2, 1),
            2: lambda : self.innitValues(5, [[0, 0 ,0 , 0, 1/5], [0, 0, 0 , 1/5, 0], [0, 0 ,1/5 , 0, 0], [0, 0 , 1/5 , 0, 0], [1/5, 0 ,0 , 0, 0]], 2, 1),
            3: lambda : self.innitValues(5, [[0, 0 ,0 , 1/5, 0], [0, 0 ,0 , 1/5, 0], [0, 0 ,1/5 , 0,0], [0, 1/5 ,0 , 0, 0], [0, 1/5 ,0 , 0, 0]], 2, 1),
            4: lambda : self.innitValues(5, [[0, 0 ,1/5 , 0, 0], [0, 0 ,1/5 , 0, 0], [0, 0 ,1/5 , 0, 0], [0, 0 , 1/5 , 0, 0], [0, 0 , 1/5 , 0, 0]], 2, 1),
            5: lambda : self.innitValues(5, [[0, 1/5 ,0 , 0, 0], [0, 1/5 ,0 , 0, 0], [0, 0 ,1/5 , 0, 0], [0, 0 ,0 , 1/5, 0], [0, 0 ,0 , 1/5, 0]], 2, 1),
            6: lambda : self.innitValues(5, [[1/5, 0 ,0 , 0, 0], [0, 1/5 ,0 , 0, 0], [0, 0 ,1/5 , 0, 0], [0, 0 ,0 , 1/5, 0], [0, 0 ,0 , 0, 1/5]], 2, 1),
            7: lambda : self.innitValues(5, [[0, 0 ,0 , 0, 0], [1/5, 1/5 ,0 , 0, 0], [0, 0 ,1/5 , 0, 0], [0, 0 ,0 , 1/5, 1/5], [0, 0 ,0 , 0, 0]], 2, 1),
        }
        options[i]()


    def calcGausian(self, x, y, sigma):
        ## TODO: only recalculate the exponent 
        muiltiplier = 1 / (2 * pi * power(sigma, 2))
        exponent = (- (power(x, 2) + power(y , 2))/(2 * power(sigma,  2)))
        distribution = muiltiplier * exp(exponent) 

        return distribution

def main() :
    colorImg = Image.open(file)
    greyImg = ImageOps.grayscale(colorImg)
    
    myEdgeFilter(greyImg, sigma) 
    colorImg.close()

## Takes a PIL image, and does the magic on it given sigma 
def myEdgeFilter(img0, sigma):
    size = img0.size
    data = asarray(img0)

    kernel = Kernel()
    
    ##Apply gausian blurr to greyscale image.  
    kernel.innitGausian(sigma)
    dataGausianFiltered = convolve(data, kernel, applyKernel)
    dataGausianFiltered = dataGausianFiltered
    dataGausianFilteredImage = Image.fromarray(dataGausianFiltered, mode="L")
    #print("Gausian")
    #print(dataGausianFiltered)
    
    ## Convole smoothed image with x sobel kernel 
    #print("Sobel X")
    kernel.innitSobelX()
    dataGradientX = convolve(dataGausianFiltered, kernel, applyKernel)
    #print(dataGradientX)
    #print("Sobel Y")
    kernel.innitSobelY()
    dataGradientY = convolve(dataGausianFiltered, kernel, applyKernel)
    #print(dataGradientY)

    ## Calculate magnitude of gradient at each point
    dataMagnitude = sobelMagnitude(dataGradientX, dataGradientY)
    dataMagnitude = normalizeTo255(dataMagnitude)
    dataMagnitudeImage = Image.fromarray(dataMagnitude.astype(numpy.uint8), mode="L")
    print("magnitude")
    print(dataMagnitude)
    dataMagnitudeImage.show()

    #Calculate direction of magnitude in degrees
    dataMagnitudeDirections = sobelDirection(dataGradientX, dataGradientY)
    dataMagnitudeDirectionsImage = Image.fromarray(normalizeTo255(dataMagnitudeDirections).astype(numpy.uint8), mode="L")
    dataMagnitudeDirectionsImage.show()
    print("Directions")
    print(dataMagnitudeDirections)

    #Calculate the threshholds using the magnitude and direction
    dataEdgesDetected = getEdges(dataMagnitude, dataMagnitudeDirections, threshhold)
    dataEdgeDetectedImage = Image.fromarray(dataEdgesDetected.astype(numpy.uint8), mode="L")
    print("Edges")
    print(dataEdgesDetected.astype(numpy.uint8))
    dataEdgeDetectedImage.show()
    #print("Edges")
    #print(dataEdgesDetected)

    img1 = dataEdgeDetectedImage

    stickFilterData = convolve(dataMagnitude, kernel, stickFilter)
    stickFilterData = normalizeTo255(stickFilterData)
    stickFilterDataImage = Image.fromarray(stickFilterData.astype(numpy.uint8), mode="L")
    print("Stick filter on mag")
    print(stickFilterData)
    stickFilterDataImage.show()

    dataEdgesDetectedStick = getEdges(stickFilterData, dataMagnitudeDirections, threshhold2)
    dataEdgesDetectedStickImage = Image.fromarray(dataEdgesDetectedStick.astype(numpy.uint8), mode="L")
    print("Stick filter on mag edges")
    print(dataEdgesDetectedStick)
    dataEdgesDetectedStickImage.show()

    dataMagnitudeImage.save(f"1-Magnitude-{file}")
    dataMagnitudeDirectionsImage.save(f"2-Directions-{file}")
    dataEdgeDetectedImage.save(f"3-Edges-{file}")
    stickFilterDataImage.save(f"4-SticksOnMag-{file}")
    dataEdgesDetectedStickImage.save(f"5-EdgesOnSticks-{file}")
    return img1

## Takes an imgArr representing the image, convolves it with the kernel. Returns a new imgArr with convolved values. 
def convolve(imgArr, kernel, operator):
    height, width = imgArr.shape
    ##2D array of type 8 bit int 
    convolvedImg = array([[0] * width for k in range(height)])
    ## Iterate through every pixel in image, apply operation to that pixel
    for i in range(height):
        for j in range(width):
            convolution = operator(imgArr, kernel, i, j)
            convolvedImg[i][j] = convolution
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
    dataMagnitudeGradient = array([[0] * width for i in range(height)])
    for i in range(height):
        for j in range(width):
            dataMagnitudeGradient[i][j] = sqrt(power(gradientX[i][j], 2) +  power(gradientY[i][j], 2))
    return dataMagnitudeGradient

## Creates an array of the directions of the gradient. 
def sobelDirection(gradientX, gradientY):
    height, width = gradientX.shape
    dataSobelDirection = array([[0] * width for i in range(height)])
    for i in range(height):
        for j in range(width):
            dataSobelDirection[i][j] = arctan2(gradientY[i][j], gradientX[i][j])
    return dataSobelDirection

## Gets the direcetion of the neighbours
def sobelDirectionToRegion(direction):
    direction = degrees(direction) % 180
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
def getEdges(gradientMagnitude, gradientDirections, threshLevel):
    height, width = gradientMagnitude.shape
    dataSobelEdges = array([[0] * width for i in range(height)])
    for i in range(height):
        for j in range(width):
            neighbour1, neighbour2 = getThreshHoldNeighbours(i, j, gradientDirections[i][j])
            threshhold = isThreshHold(gradientMagnitude, (i, j), neighbour1, neighbour2)
            if(not threshhold):
                dataSobelEdges[i][j] = 0 
            else:
                dataSobelEdges[i][j] = getThresholdLevel(threshLevel, gradientMagnitude[i][j]) 
    return dataSobelEdges

def getThresholdLevel(threshLevel, value):
    if(value < threshLevel): return 0
    #elif(value < threshhold2): return 127
    else: return 255

## Returns the coordinates of the theshhold neighbours 
def getThreshHoldNeighbours(i, j, gradientDirection):
    gradientDirection = sobelDirectionToRegion(gradientDirection)
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
def isThreshHold(gradientMagnitude, center, neighbour1, neighbour2):
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

def stickFilter(imgArr, kernel, i, j):
    ## Get Average of 5x5 neighbourhood
    kernel.innitAverage(5)
    convolutionSum = 0
    for u in range(- kernel.k, kernel.k + 1):
        for v in range(- kernel.k, kernel.k + 1) :
            ## Negative out of bounds check
            if(i - u >= 0 and j - v >= 0):
                try:
                    convolutionSum += imgArr[i - u][j - v]
                ## Positive out of bounds catch
                except IndexError:
                    continue
    neighbourHoodAverage = convolutionSum/25

    averageStick = [0, 0, 0, 0, 0, 0, 0, 0]
    for k in range(8):
        kernel.innitStickFilter(k)
        
        for u in range(- kernel.k, kernel.k + 1):
            for v in range(- kernel.k, kernel.k + 1) :
                if(i - u >= 0 and j - v >= 0):
                    try:
                        averageStick[k] += imgArr[i - u][j - v] * kernel.get(u, v)
                    except IndexError:
                        averageStick[k] += 0
        averageStick[k] = averageStick[k] / 5
    
    differences = [ abs(neighbourHoodAverage - l) for l in averageStick]
    maxDiff = max(differences)
    maxIndex = differences.index(maxDiff)
    return averageStick[maxIndex]

def normalizeTo255(imgArr):
    maxVal = numpy.max(imgArr)

    height, width = imgArr.shape
    normalized255Img = array([[0] * width for i in range(height)])

    for i in range(height):
        for j in range(width):
            normalized255Img[i][j] = imgArr[i][j] / maxVal * 255
    return normalized255Img
if __name__ == "__main__":
    main() 
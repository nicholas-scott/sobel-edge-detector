

sobelXKernel = ""
sobelYKernel = ""

def main() :
    return

def myEdgeFilter(img0, sigma):
    img1 = ""
    
    ## Get kernel size
    smoothKernel = getSmoothKernel(sigma)

    ##Convolve image to smooth it 
    smoothImg = convolve(img0, smoothKernel)

    ## Get x and y sobel gradients of smoothed image 
    gradientXImg = convolve(smoothImg, sobelXKernel)
    gradientYImg = convolve(smoothImg, sobelYKernel)

    ## Do the rest lol 

    return img1

def convolve(img):
    return 

def getSmoothKernel(sigma):
    ##(e.g., hsize = 2 * ceil(3 * sigma) + 1)
    return

if __name__ == "__main__":
    main()
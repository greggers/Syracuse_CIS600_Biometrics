import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the iris image by converting it to grayscale, normalizing, and applying Gaussian blur.
    
    Parameters:
    image_path (str): The path to the iris image file.
    
    Returns:
    np.ndarray: The preprocessed image.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the image to enhance contrast
    normalized_image = cv2.equalizeHist(image)

    # Apply Gaussian Blur to reduce noise and smooth the image
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    return blurred_image

def segment_iris(image):
    """
    Segment the iris by detecting the iris boundaries using edge detection and Hough Circle Transform.
    
    Parameters:
    image (np.ndarray): The preprocessed iris image.
    
    Returns:
    np.ndarray: The image with the detected iris boundaries.
    """
    # Apply edge detection to find edges in the image
    edges = cv2.Canny(image, 100, 200)

    # Apply Hough Circle Transform to detect circles (iris boundaries) in the image
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40, param1=150, param2=80, minRadius=20, maxRadius=None)

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))

        # Loop through each of the detected circles
        for i in circles[0, :] :
            # Draw the outer boundary of the circle
            cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    return image, circles

def rubber_mat_normalization(image, circles):
    """
    Apply rubber mat normalization to the segmented iris.
    
    Parameters:
    image (np.ndarray): The preprocessed iris image after segmentation.
    
    Returns:
    np.ndarray: The normalized iris image using rubber mat technique.
    """
    # Find the center of the circle
    x, y, r = circles[0][1]
    
    # Create a mask for the inner circle
    mask = np.zeros((image.shape), dtype=np.uint8)
    cv2.circle(mask, (x, y), int(r*1.5), 255, -1)

    # Apply thresholding to find pixels outside the circle
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the outer circle boundaries
    _, contours, _ = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(opened_image, (x, y), (x+w, y+h), 255, -1)

    # Create a mask for the outer circle boundaries
    outer_mask = np.zeros((image.shape), dtype=np.uint8)
    cv2.drawContours(outer_mask, [contour], -1, 255, 1)

    # Normalize the image using rubber mat technique
    normalized_image = np.multiply(image, (mask + outer_mask) / 255.0)

    return normalized_image

def gabor_feature_encoding(image):
    """
    Apply Gabor filter to extract features from the iris image.
    
    Parameters:
    image (np.ndarray): The preprocessed iris image after rubber mat normalization.
    
    Returns:
    np.ndarray: The image after applying the Gabor filter.
    """
    # Create Gabor kernel (filter)
    ksize = 31
    sigma = 4.0
    theta = np.pi/4
    lambd = 10.0
    gamma = 0.5
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, ktype=cv2.CV_32F)
    
    # Apply the Gabor filter to the image
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    
    return filtered_image

def main():
    """
    Main function to demonstrate the Daugman Iris Identification Algorithm steps.
    """
    image_path = 'Module_05_Iris_Identification/iris2.jpg'

    # Preprocess the image
    processed_image = preprocess_image(image_path)
    cv2.imshow('Processed Iris Image', processed_image)
    cv2.waitKey(0)

    # Segment the iris
    segmented_image, circles = segment_iris(processed_image)
    cv2.imshow('Detected Iris', segmented_image)
    cv2.waitKey(0)

    # Apply rubber mat normalization
    normalized_image = rubber_mat_normalization(processed_image, circles)
    cv2.imshow('Normalized Iris Image', normalized_image)
    cv2.waitKey(0)

    # Gabor feature encoding
    encoded_image = gabor_feature_encoding(normalized_image)
    cv2.imshow('Gabor Filtered Image', encoded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
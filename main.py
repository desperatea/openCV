import cv2
import ntpath
from matplotlib import pyplot as plt

image_directory = 'img/'
output_directory = 'output/'


def view_image(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def view_cropped():
    image = cv2.imread(image_directory + 'dog1.jpg')
    cropped = image[40:500, 500:2000]

    view_image(image, 'Before Cropped')
    view_image(cropped, 'After Cropped')

    cv2.imwrite(output_directory + 'dog1_cropped.jpg', cropped)


def view_resized():
    image = cv2.imread(image_directory + 'dog2.jpg')

    scale_percent = 20  # Процент от изначального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    view_image(image, "Before changing size")
    view_image(resized, "After changing size to 20 %")

    cv2.imwrite(output_directory + 'dog2_resized.jpg', resized)


def view_rotated():
    image = cv2.imread(image_directory + 'dog3.jpg')

    (h, w, d) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    view_image(image, "Before")
    view_image(rotated, "After rotation")

    cv2.imwrite(output_directory + 'dog3_rotated.jpg', rotated)


def view_gray_and_threshold():
    image = cv2.imread(image_directory + 'dog4.jpg')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)

    view_image(image, "Before")
    view_image(gray_image, "Gray")
    view_image(threshold_image, "Threshold")

    cv2.imwrite(output_directory + 'dog4_gray.jpg', gray_image)
    cv2.imwrite(output_directory + 'dog4_threshold.jpg', threshold_image)


def view_blurred():
    image = cv2.imread(image_directory + 'dog5.jpg')
    blurred = cv2.GaussianBlur(image, (51, 51), 0)

    view_image(image, "Before")
    view_image(blurred, "After")

    cv2.imwrite(output_directory + 'dog5_blurred.jpg', blurred)


def view_rectangle():
    image = cv2.imread(image_directory + 'dog8.jpeg')
    output = image.copy()
    cv2.rectangle(output, (250, 60), (450, 250), (0, 255, 255), 10)

    view_image(image, "Before")
    view_image(output, "After")

    cv2.imwrite(output_directory + 'dog8_rectangle.jpg', output)


def view_line():
    image = cv2.imread(image_directory + 'dog6.jpg')
    output = image.copy()
    cv2.line(output, (220, 60), (220, 550), (0, 255, 255), 3)

    view_image(image, "Before")
    view_image(output, "After")

    cv2.imwrite(output_directory + 'dog6_line.jpg', output)


def view_text():
    image = cv2.imread(image_directory + 'dog7.jpg')
    output = image.copy()
    cv2.putText(output, "I'm crying in my soul", (350, 800), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    view_image(image, "Before")
    view_image(output, "After")

    cv2.imwrite(output_directory + 'dog6_text.jpg', output)


def find_faces(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )

    faces_detected = "Face detected: " + format(len(faces))
    print(faces_detected)

    # Рисуем квадраты вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return image, faces_detected


def view_faces():
    image = cv2.imread(image_directory + 'faces1.jpg')
    output, faces_detected = find_faces(image)
    view_image(output, faces_detected)
    cv2.imwrite(output_directory + 'faces1.jpg', output)

    image = cv2.imread(image_directory + 'faces2.jpg')
    output, faces_detected = find_faces(image)
    view_image(output, faces_detected)
    cv2.imwrite(output_directory + 'faces2.jpg', output)

    image = cv2.imread(image_directory + 'faces3.jpg')
    output, faces_detected = find_faces(image)
    view_image(output, faces_detected)
    cv2.imwrite(output_directory + 'faces3.jpg', output)


if __name__ == '__main__':
    view_cropped()
    view_resized()
    view_rotated()
    view_gray_and_threshold()
    view_blurred()
    view_rectangle()
    view_line()
    view_text()
    view_faces()

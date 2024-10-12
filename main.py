import cv2

face_ref = cv2.CascadeClassifier("face.xml")

cam = cv2.VideoCapture(0)

def face_detect(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=3)
    return faces

def drawer(frame):
    for x, y, w, h in face_detect(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 230, 20), 2)

def close_window():
    cam.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = cam.read()
        drawer(frame)
        cv2.imshow("Face detection", frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
          close_window()

if __name__ == '__main__':
    main()
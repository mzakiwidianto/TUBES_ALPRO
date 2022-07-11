from scipy.spatial import distance
from imutils import face_utils
import playsound
import imutils
import dlib
import cv2

# Fungsi untuk menghitung aspek rasio mata(EAR)
def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2 * c)
    return ear

# Fungsi untuk menyalakan alarm
def alarm():
    playsound.playsound("tone.mp3")

# Batas EAR
ear_treshold = 0.18
# Batas jumlah frame setelah melewati batas EAR
frame_counter_tresh = 20
frame_counter = 0

# Menggunakan modul pendeteksi yang sudah dilatih dari file (.xml/.dat)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Mengamil index landmark wajah pada bagian mata kanan dan kiri
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Memulai pengambilan gambar
cap = cv2.VideoCapture(1)

# Loop yang dijalankan pada setiap frame
while True:
    # Mengambil data dari kamera, lalu mengubahnya menjadi grayscale
    ret, frame = cap.read()
    height = int(cap.get(4))
    width = int(cap.get(3))
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mendeteksi wajah pada frame grayscale
    sbj = detect(gray, 0)

    # Loop pendeteksi wajah
    for subject in sbj:
        # Konversi koordinat titik wajah menjadi NumPy array
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        # Mendapatkan koordinat mata kanan dan kiri
        left = shape[lStart:lEnd]
        right = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(left)
        rightEAR = eye_aspect_ratio(right)
        ear = (leftEAR + rightEAR) / 2.0
        # Menggambarkan garis di sekeliling mata berdasarkan koordinat
        leftHull = cv2.convexHull(left)
        rightHull = cv2.convexHull(right)
        cv2.drawContours(frame, [leftHull], -1, (255,255,255), 1)
        cv2.drawContours(frame, [rightHull], -1, (255,255,255), 1)
        # Pendeteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h ), (0, 255, 0), 2)

            # Mengubah kotak deteksi menjadi merah
            if frame_counter >= frame_counter_tresh:
                cv2.rectangle(frame, (x, y), (x + w, y + h ), (0, 0, 255), 2)

        # Membatasi ear dengan ear_treshold
        if ear < ear_treshold:
            # Menghitung frame jika melewati batas/tresholdq
            frame_counter += 1
            print(frame_counter, ear)

            # Kondisi menambahkan peringatan berupa text
            if frame_counter >= frame_counter_tresh:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "************************ HEY BANGUN ************************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Kondisi untuk menyalakan alarm
                if frame_counter > 23:
                    print("====!!!warning!!!====")
                    alarm()
        else:
            frame_counter = 0
    # Menampilkan gambar yang sudah diproses
    cv2.imshow("Pendeteksi Kesadaran Pengendara", frame)
    key = cv2.waitKey(1) & 0xFF

    # Kondisi untuk keluar dari program
    if key == ord("q"):
        break
# Membersihkan sisa dari jalannya program
cv2.destroyAllWindows()
cap.release()
import cv2
import face_recognition
import mysql.connector
from datetime import datetime
import csv
import threading
from flask import Flask, render_template, Response

app = Flask(__name__)

# Inisialisasi kamera
video_capture = cv2.VideoCapture(0)  # 0 untuk kamera laptop, 1 untuk kamera eksternal
video_capture.set(3, 1080)  # Atur lebar
video_capture.set(4, 720)  # Atur tinggi

# Koneksi ke database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="face"
)

# Cursor
cursor = db.cursor()

# Ambil data pengguna dari database
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

# Inisialisasi list wajah yang dikenali
known_face_names = []
known_face_encodings = []

for r in rows:
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("images/" + r[1]))[0])  # Proses encoding
    known_face_names.append(r[0])  # Nama orang

people = known_face_names.copy()  # Simpan nama orang

# Ambil tanggal sekarang
current_date = datetime.now().strftime("%Y-%m-%d")

# Buat file CSV
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Set untuk melacak wajah yang telah terdeteksi
detected_people = set()

# Fungsi untuk membaca frame dari kamera dan deteksi wajah
def process_frames():
    global video_capture, known_face_encodings, known_face_names, detected_people

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Gagal membaca frame, keluar...")
            break

        frame = cv2.flip(frame, 1)  # Mirror

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengubah dari BGR (format openCV) ke RGB (format face_recognition)

        face_locations = face_recognition.face_locations(rgb_frame)  # Mendeteksi lokasi wajah dalam gambar
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # Mengambil encoding dari wajah yang terdeteksi

        # Proses pengenalan wajah dan tampilan
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)  # Membandingkan encoding wajah yang terdeteksi dengan encoding wajah yang sudah dikenali
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if name != "Unknown":  # Jika wajah dikenali
                if name not in detected_people:  # Hanya tulis masuk jika wajah belum terdeteksi
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    detected_people.add(name)  # Tambahkan wajah ke set deteksi

                # Tampilkan kotak dan nama wajah
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def start_thread():
    thread = threading.Thread(target=process_frames)
    thread.daemon = True
    thread.start()

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '_main_':
    start_thread()
    app.run(host='0.0.0.0', port=5000, debug=True)

# Tutup koneksi dan file
video_capture.release()
cv2.destroyAllWindows()
f.close()
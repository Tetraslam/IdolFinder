import face_recognition
# Loop through loads and encodings to enable UI to detect smoothly
known_image = face_recognition.load_image_file("./training/nayeon.jpg")
unknown_image = face_recognition.load_image_file("./validation/valLisa.jpg")

nayeon_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([nayeon_encoding], unknown_encoding)
print(results)
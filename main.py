import face_recognition
# Loop through loads and encodings to enable UI to detect smoothly
nayeon = face_recognition.load_image_file("./training/nayeon.jpg")
jeongyeon = face_recognition.load_image_file("./training/jeongyeon.jpg")

unknown_image = face_recognition.load_image_file("./validation/valJeongyeon.jpg")

nayeon_encoding = face_recognition.face_encodings(nayeon)[0]
jeongyeon_encoding = face_recognition.face_encodings(jeongyeon)[0]

unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([nayeon_encoding], unknown_encoding)
print(*results)
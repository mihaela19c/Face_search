import cv2 
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name = "buffalo_l")
face_app.prepare(ctx_id = -1)

img = cv2.imread("photos/photo_1.jpg")
faces = face_app.get(img)

print("Face found:", len(faces))
for i, face in enumerate(faces):
    print("Face ", i)
    print(" Box:", face.bbox)
    print(" Embedding lenght:", len(face.normed_embedding))
    print(" First 5 numbers:", face.normed_embedding[:5])

    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

cv2.imwrite("output_boxes.jpeg", img)
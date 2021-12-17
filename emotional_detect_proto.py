from fer import FER
import matplotlib.pyplot as plt

#%matplotlib inline
test_image_one = plt.imread("/Desktop/abba.png")
detector = FER(mtcnn=True)
captured_emotions = detector.detect_emotions(test_image_one)
print(captured_emotions)
plt.imshow(test_image_one)
primary_emotion, emotion_score = detector.top_emotion(test_image_one)
print(primary_emotion, emotion_score)
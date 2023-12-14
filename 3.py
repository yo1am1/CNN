from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions

# Завантаження зображень
image_paths = [
    "./test_images/dog.png",
    "./test_images/car.png",
    "./test_images/person.png",
]

for path in image_paths:
    # Завантаження та підготовка зображення
    image = load_img(path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(
        image_array.reshape(
            (1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
        )
    )

    # Вибір попередньо натренованої моделі (можна використовувати і ResNet50)
    model = VGG16(include_top=True, weights="imagenet")

    # Передбачення класу
    y_pred = model.predict(image_array)

    # Декодування та вивід результату
    labels = decode_predictions(y_pred)
    print(f"Predictions for {path}: {labels[0][0][1]} ({labels[0][0][2] * 100 :.2f} %)")

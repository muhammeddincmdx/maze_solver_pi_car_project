# --- Kütüphaneler ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- Parametreler ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

# --- Veri yolları ---
BASE_PATH = 'maze'
IMAGE_PATH = os.path.join(BASE_PATH, '')
MASK_PATH = os.path.join(BASE_PATH, 'masks')

# --- Eğitim parametreleri ---
EPOCHS = 25
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1

# --- Görüntü ve Maske Dosyalarını Yükleme ---
try:
    image_files = sorted([
        os.path.join(IMAGE_PATH, f)
        for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    mask_files = sorted([
        os.path.join(MASK_PATH, f)
        for f in os.listdir(MASK_PATH)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
except FileNotFoundError:
    print(f"Hata: '{IMAGE_PATH}' veya '{MASK_PATH}' klasör yolu bulunamadı.")
    print("Lütfen IMAGE_PATH ve MASK_PATH değişkenlerini kontrol edin.")
    exit()

if len(image_files) != len(mask_files):
    print(f"Uyarı: Görüntü sayısı ({len(image_files)}) ile maske sayısı ({len(mask_files)}) eşleşmiyor!")
    exit()

if len(image_files) == 0:
    print(f"Hata: '{IMAGE_PATH}' veya '{MASK_PATH}' klasöründe uygun resim dosyası bulunamadı.")
    exit()

print(f"Toplam {len(image_files)} adet görüntü/maske çifti bulundu.")

# --- Eğitim ve Doğrulama Seti Ayırma ---
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_files, mask_files, test_size=VALIDATION_SPLIT, random_state=42
)

print(f"Eğitim seti: {len(train_img_paths)}, Doğrulama seti: {len(val_img_paths)}")


# --- Görüntü ve Maske Yükleme Fonksiyonu ---
def load_and_preprocess_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=IMG_CHANNELS)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=IMG_CHANNELS)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask


# --- Dataset Oluşturma Fonksiyonu ---
def create_dataset(img_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    buffer_size = len(img_paths) if len(img_paths) > 0 else BATCH_SIZE * 10
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Datasetleri Oluştur ---
train_dataset = create_dataset(train_img_paths, train_mask_paths, BATCH_SIZE)
val_dataset = create_dataset(val_img_paths, val_mask_paths, BATCH_SIZE)

print("tf.data pipeline oluşturuldu.")


# --- Basit U-Net Modeli ---
def build_unet_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c3])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    u7 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c2])
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    u8 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c1])
    c8 = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# --- Model Oluşturma ve Derleme ---
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = build_unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Model Eğitimi ---
print("\nEğitim Başlatılıyor...")
# model_checkpoint = keras.callbacks.ModelCheckpoint('best_maze_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
    # callbacks=[model_checkpoint]
)

print("Eğitim Tamamlandı.")


# --- Model Kaydetme ---
final_model_filename = 'final_mazee_segmentation_unet_model.h5'
model.save(final_model_filename)
print(f"Eğitilmiş son model '{final_model_filename}' olarak kaydedildi.")

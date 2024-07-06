import os
import time
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Kullanıcıdan epoch ve batch size değerlerini al
epochs = int(input("Kaç epoch olsun: "))
batch_size = int(input("Batch size: "))

# Yeni Veri Seti Parametreleri
new_label_map = {1: "new_class1", 2: "new_class2", 3: "new_class3"}  # Yeni veri seti için etiketler

new_train_images_dir = "new_dataset/train/images"
new_train_annotations_dir = "new_dataset/train/annotations"
new_val_images_dir = "new_dataset/validation/images"
new_val_annotations_dir = "new_dataset/validation/annotations"
new_test_images_dir = "new_dataset/test/images"
new_test_annotations_dir = "new_dataset/test/annotations"

# Yeni Veri Setini Yükleme
new_train_data = object_detector.DataLoader.from_pascal_voc(new_train_images_dir, new_train_annotations_dir, label_map=new_label_map)
new_validation_data = object_detector.DataLoader.from_pascal_voc(new_val_images_dir, new_val_annotations_dir, label_map=new_label_map)
new_test_data = object_detector.DataLoader.from_pascal_voc(new_test_images_dir, new_test_annotations_dir, label_map=new_label_map)

# Eğitilmiş Modeli Yükleme
model_spec = object_detector.EfficientDetLite0Spec()
model_dir = "./models"
saved_model_path = os.path.join(model_dir, "dosya.tflite")

# Modelin TFLite dosyasından yeniden yüklenmesi
interpreter = tf.lite.Interpreter(model_path=saved_model_path)
interpreter.allocate_tensors()

# Fine-Tuning İçin Mevcut Modeli Kullanarak Yeni Modeli Oluşturma
model = object_detector.create(new_train_data, model_spec, new_validation_data, epochs, batch_size, train_whole_model=True)

# Yeni Modeli Değerlendirme
print("Yeni model değerlendiriliyor...")
model.evaluate(new_test_data)

# Yeni Modeli TensorFlow Lite'a Dönüştürme
fine_tuned_tflite_filename = "fine_tuned_model.tflite"
fine_tuned_labels_filename = "fine_tuned_labels.txt"

print("Yeni model TensorFlow Lite formatına dönüştürülüyor...")
model.export(export_dir=".", tflite_filename=fine_tuned_tflite_filename, label_filename=fine_tuned_labels_filename, export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

# Dönüştürülen Yeni Modeli Değerlendirme
print("Dönüştürülen yeni model değerlendiriliyor...")
model.evaluate_tflite(fine_tuned_tflite_filename, new_test_data)

import os
import shutil
import numpy as np
import tensorflow as tf


execution_path = os.getcwd()
base_dir = os.path.join(execution_path, 'C:/Users/AmbruszkiewM/PycharmProjects/klasyfikator_tf/dane')

class_1 = 'bicycle'
class_2 = 'motorcycle'
train_ratio = 0.7
valid_ratio = 0.2
data_dir= r'./images'

raw_no_of_files = {}
classes = [class_1, class_2]

number_of_samples = [(dir, len(os.listdir(os.path.join(base_dir, dir)))) for dir in classes]
print(number_of_samples)

if not os.path.exists(data_dir): os.mkdir(data_dir)

#katalogi dla zbiorów

train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

train_class_1_dir = os.path.join(train_dir, class_1)
valid_class_1_dir = os.path.join(valid_dir, class_1)
test_class_1_dir = os.path.join(test_dir, class_1)

train_class_2_dir = os.path.join(train_dir, class_2)
valid_class_2_dir = os.path.join(valid_dir, class_2)
test_class_2_dir = os.path.join(test_dir, class_2)

for dir in (train_dir, valid_dir, test_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_1_dir, valid_class_1_dir, test_class_1_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

for dir in (train_class_2_dir, valid_class_2_dir, test_class_2_dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

print('[info] Wczytywanie nazw plików...')
class_1_names = os.listdir(os.path.join(base_dir, class_1))
class_2_names = os.listdir(os.path.join(base_dir, class_2))

print('[info] Walidacja poprawności nazw...')
class_1_names = [fname for fname in class_1_names if fname.split('.')[1].lower() in ['jpg']]
class_2_names = [fname for fname in class_2_names if fname.split('.')[1].lower() in ['jpg']]

# shuffle nazw plików

np.random.shuffle(class_1_names)
np.random.shuffle(class_2_names)

print(f'[info] Liczba obrazów w zbiorze {class_1}: {len(class_1_names)}')
print(f'[info] Liczba obrazów w zbiorze {class_2}: {len(class_2_names)}')

train_index_class_1 = int(train_ratio * len(class_1_names))
valid_index_class_1 = train_index_class_1 + int(valid_ratio * len(class_1_names))

train_index_class_2 = int(train_ratio * len(class_2_names))
valid_index_class_2 = train_index_class_2 + int(valid_ratio * len(class_2_names))


print('[info] Kopiowanie plików do katalogów docelowych...')
for i, fname in enumerate(class_1_names):
    if i <= train_index_class_1:
        src = os.path.join(base_dir, class_1, fname)
        dst = os.path.join(train_class_1_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_1 < i <= valid_index_class_1:
        src = os.path.join(base_dir, class_1, fname)
        dst = os.path.join(base_dir, class_1, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_1 < i <= len(class_1_names):
        src = os.path.join(base_dir, class_1, fname)
        dst = os.path.join(test_class_1_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(class_2_names):
    if i <= train_index_class_2:
        src = os.path.join(base_dir, class_2, fname)
        dst = os.path.join(train_class_2_dir, fname)
        shutil.copyfile(src, dst)
    if train_index_class_2 < i <= valid_index_class_1:
        src = os.path.join(base_dir, class_2, fname)
        dst = os.path.join(base_dir,class_2, fname)
        shutil.copyfile(src, dst)
    if valid_index_class_2 < i <= len(class_2_names):
        src = os.path.join(base_dir, class_2, fname)
        dst = os.path.join(test_class_2_dir, fname)
        shutil.copyfile(src, dst)

print(f'[info] Liczba obrazów dla klasy {class_1} w zbiorze treningowym: {len(os.listdir(train_class_1_dir))}')
print(f'[info] Liczba obrazów dla klasy {class_1} w zbiorze walidacyjnym: {len(os.listdir(valid_class_1_dir))}')
print(f'[info] Liczba obrazów dla klasy {class_1} w zbiorze testowym: {len(os.listdir(test_class_1_dir))}')

print(f'[info] Liczba obrazów dla klasy {class_2} w zbiorze treningowym: {len(os.listdir(train_class_2_dir))}')
print(f'[info] Liczba obrazów dla klasy {class_2} w zbiorze walidacyjnym: {len(os.listdir(valid_class_2_dir))}')
print(f'[info] Liczba obrazów dla klasy {class_2} w zbiorze testowym: {len(os.listdir(test_class_2_dir))}')

print(tf.__version__)
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np
import pickle as pkl
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split,KFold

def labels(path):
    print(os.listdir(path))

#THIS IS FOR DOG BREED PREDICTION DATASET
def eda(datasetpath):
    labels = os.listdir(os.path.join(datasetpath, 'train'))
    print(f"The number of class labels are:-{len(labels)}")
    print(f"The Class Labels are:-{labels}")
    imagecnt = {}
    tot_imgs = 0
    for label in labels:
        label_path = os.path.join(datasetpath, 'train', label)
        imagecnt[label] = len(os.listdir(label_path))
        tot_imgs = tot_imgs + imagecnt.get(label)
    print(f"total number of images in each class are:- {imagecnt}")
    print(f"total images are:- {tot_imgs}")

    # Original bar plot
    plt.figure(figsize=(85, 5))
    plt.bar(imagecnt.keys(), imagecnt.values())
    plt.xlabel('Dog Breeds')
    plt.xticks(rotation=90)
    plt.ylabel('Image Count')
    plt.title("Class Distribution of Dog Breeds")
    plt.show()

    # Plotting random images from different breeds
    plt.figure(figsize=(10, 5))
    random_range = random.sample(range(len(labels)), 70)
    for counter, random_index in enumerate(random_range, 70):
        selected_class_name = labels[random_index]
        img_files_names_list = os.listdir(datasetpath + '/train/' + selected_class_name)
        selected_img_file_name = random.choice(img_files_names_list)
        selected_class_path = os.path.join(datasetpath, 'train', selected_class_name)
        bgr_img = cv2.imread(os.path.join(selected_class_path, selected_img_file_name), 1)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb_img, selected_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        plt.imshow(rgb_img)
        plt.axis('off')
    plt.title("Random Samples of Dog Breeds")
    plt.show()

    # Additional Data Analysis Plots

    # Pie chart showing the distribution of classes
    plt.figure(figsize=(8, 8))
    class_counts = list(imagecnt.values())
    class_labels = list(imagecnt.keys())
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution (Pie Chart)')
    plt.axis('equal')
    plt.show()

    # Histogram of image sizes (height and width)
    image_sizes = []
    for label in labels:
        label_path = os.path.join(datasetpath, 'train', label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            image_sizes.append((height, width))

    heights, widths = zip(*image_sizes)

    # Plotting histograms for image dimensions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=20, color='blue', alpha=0.7)
    plt.title("Distribution of Image Heights")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=20, color='orange', alpha=0.7)
    plt.title("Distribution of Image Widths")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot average image size per class
    avg_sizes = {}
    for label in labels:
        label_path = os.path.join(datasetpath, 'train', label)
        class_heights = []
        class_widths = []
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            class_heights.append(height)
            class_widths.append(width)
        avg_sizes[label] = (np.mean(class_heights), np.mean(class_widths))

    avg_heights = [avg_sizes[label][0] for label in labels]
    avg_widths = [avg_sizes[label][1] for label in labels]

    # Plotting average image sizes per class
    plt.figure(figsize=(15, 5))
    plt.bar(labels, avg_heights, color='blue', alpha=0.7, label='Average Height')
    plt.bar(labels, avg_widths, color='orange', alpha=0.7, label='Average Width')
    plt.xticks(rotation=90)
    plt.xlabel("Dog Breeds")
    plt.ylabel("Average Dimension (pixels)")
    plt.title("Average Image Sizes (Height and Width) per Class")
    plt.legend(loc='upper right')
    plt.show()

def datapreprocessing(datasetpath):
    print('Datapreprocessing In progress...')
    img_height = 224
    img_width = 224
    train_targets = os.listdir(os.path.join(datasetpath,'train'))
    test_targets = os.listdir(os.path.join(datasetpath, 'test'))
    preprocess_data_file_path = "DogBreedPreprocessedData.pkl"
    load_preprocessed_data_from_file = input("Do you want to load preprocessed Data from File(yes/no)?")
    if load_preprocessed_data_from_file.lower()=='no':
        x_train = []
        y_train = []
        print("resizing,normalizing and appending the images to list")
        for target in train_targets:
            img_files = os.listdir(os.path.join(datasetpath,'train',target))
            print(f"working on {target} images")
            for img in img_files:
                img = cv2.imread(os.path.join(datasetpath,'train',target,img))
                resized_img = cv2.resize(img,(img_height,img_width))
                normalized_img = resized_img/255.0
                x_train.append(normalized_img)
                y_train.append(target)
        print(f'---------------')
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_train = to_categorical(y_train,num_classes=70)
        print(y_train)
        print(f'Successfully encoded Train Labels')

        x_test = []
        y_test = []
        print("resizing,normalizing and appending the images to list")
        for target in test_targets:
            img_files = os.listdir(os.path.join(datasetpath, 'test', target))
            print(f"working on {target} images")
            for img in img_files:
                img = cv2.imread(os.path.join(datasetpath, 'test', target, img))
                resized_img = cv2.resize(img, (img_height, img_width))
                normalized_img = resized_img / 255.0
                x_test.append(normalized_img)
                y_test.append(target)
        print(f'---------------')
        labelencoder = LabelEncoder()
        y_test = labelencoder.fit_transform(y_test)
        y_test = to_categorical(y_test,num_classes=70)
        print(y_test)
        print(f'Successfully encoded Test Labels')

        print('converting to tensors')
        x_train = tf.convert_to_tensor(x_train,dtype='float32')
        y_train = tf.convert_to_tensor(y_train,dtype='float32')
        x_test = tf.convert_to_tensor(x_test,dtype='float32')
        y_test = tf.convert_to_tensor(y_test,dtype='float32')
        print('tensor conversion done')
        with open(preprocess_data_file_path, 'wb') as file:
            pkl.dump((x_train,x_test,y_train,y_test), file)
        print('PreProcessed data saved to file Successfully')
        print(f'preprocessed data shapes are:-\n' + 'x_train shape-', np.shape(x_train), ' y_train shape-',np.shape(y_train))
        preprocessed_data_status = True
        print('Data preprocessing Completed')
        return x_train,x_test,y_train,y_test,preprocessed_data_status
    elif load_preprocessed_data_from_file=='yes':
        with open(preprocess_data_file_path,'rb') as file:
            x_train,x_test,y_train,y_test = pkl.load(file)
            print("Data Loaded Successfully from file",os.path.basename(preprocess_data_file_path))
        print(f'preprocessed data shapes and types are:-\n', 'x_train shape-', np.shape(x_train),type(x_train), ' y_train shape-',np.shape(y_train),type(y_train), '\nx_test shape-', np.shape(x_test),type(x_test), ' y_test shape-', np.shape(y_test),type(y_test))
        preprocessed_data_status = True
        return x_train,x_test,y_train,y_test,preprocessed_data_status
    else:
        print("Option Chosen Not found")
        preprocessed_data_status = False
        return preprocessed_data_status


#datapreprocessing for standford dataset
def datapreprocessing_standford_dataset(datasetpath,img_height,img_width):
    print('Datapreprocessing In progress...')
    targets = os.listdir(os.path.join(datasetpath))
    file_end = str(img_width)+"X"+str(img_height)
    preprocess_data_file_path = f"outputs/StandfordDogsBreedPreprocessedData299X299.pkl"
    load_preprocessed_data_from_file = input("Do you want to load preprocessed Data from File(yes/no)?")
    if load_preprocessed_data_from_file=='no':
        x = []
        y = []
        print("resizing,normalizing and appending the images to list")
        for target in targets:
            img_files = os.listdir(os.path.join(datasetpath,target))
            print(f"working on {target} images")
            for img in img_files:
                img = cv2.imread(os.path.join(datasetpath,target,img))
                resized_img = cv2.resize(img,(img_height,img_width))
                normalized_img = resized_img/255.0
                x.append(normalized_img)
                y.append(target)
        print(f'---------------')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True,random_state=42)
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_train = to_categorical(y_train,num_classes=120)
        y_test = labelencoder.fit_transform(y_test)
        y_test = to_categorical(y_test, num_classes=120)
        print(y_train)
        print(f'Successfully encoded Labels')
        print('converting to tensors')
        x_train = tf.convert_to_tensor(x_train,dtype='float32')
        y_train = tf.convert_to_tensor(y_train,dtype='float32')
        x_test = tf.convert_to_tensor(x_test, dtype='float32')
        y_test = tf.convert_to_tensor(y_test, dtype='float32')
        print('tensor conversion done')
        with open(preprocess_data_file_path, 'wb') as file:
            pkl.dump((x_train,x_test,y_train,y_test), file)
        print('Standford Dogs PreProcessed data saved to file Successfully')
        print(f'preprocessed data shapes are:-\n' + 'x_train shape-', np.shape(x_train), ' y_train shape-',np.shape(y_train), 'x_test shape-', np.shape(x_test), ' y_test shape-',np.shape(y_test))
        preprocessed_data_status = True
        print('Data preprocessing Completed')
        return x_train,x_test,y_train,y_test,preprocessed_data_status
    elif load_preprocessed_data_from_file=='yes':
        with open(preprocess_data_file_path,'rb') as file:
            x_train,x_test,y_train,y_test = pkl.load(file)
            print("Data Loaded Successfully from file",os.path.basename(preprocess_data_file_path))
        print(f'preprocessed data shapes and types are:-\n', 'x_train shape-', np.shape(x_train),type(x_train), ' y_train shape-',np.shape(y_train),type(y_train), '\nx_test shape-', np.shape(x_test),type(x_test), ' y_test shape-', np.shape(y_test),type(y_test))
        preprocessed_data_status = True
        return x_train,x_test,y_train,y_test,preprocessed_data_status
    else:
        print("Option Chosen Not found")
        preprocessed_data_status = False
        return preprocessed_data_status

def imagedatageneration(x_train,y_train,batch_size):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=batch_size)
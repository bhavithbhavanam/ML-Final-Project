import os
import tensorflow as tf
import tensorflow.keras as tf_keras
import numpy as np
import heapq
import seaborn as sns
import cv2
from keras.src.optimizers import SGD
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Conv2D,MaxPooling2D,Dense,InputLayer,BatchNormalization,GlobalMaxPooling2D,Flatten,Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,Precision,Recall,F1Score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.keras.applications import VGG16,VGG19,ResNet50,EfficientNetB0
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

def simplecnn(status,image_height,img_width):
        model = Sequential()
        model.add(InputLayer(input_shape=(image_height,img_width,3)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(GlobalMaxPooling2D())
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(120,activation='softmax'))
        print("Simple CNN Model Built is Successfull")
        print(model.summary())
        if status:
            model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=[Accuracy(),Precision(),Recall(),F1Score()])
            print("Simple CNN Model ready for training")
            return model
        else:
            print("Data not available to train the Simple CNN model")
            return

def vgg16(status,image_height,img_width):
    base_model = VGG16(weights='imagenet',include_top=False,input_shape=(image_height,img_width,3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))  # Another fully connected layer
    model.add(Dense(70, activation='softmax'))
    if status:
        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), F1Score()])
        print("VGG16 Model ready for training")
        print(model.summary())
        return model

def vgg19(status,image_height,img_width):
    base_model = VGG19(weights='imagenet',include_top=False,input_shape=(image_height,img_width,3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))  # Another fully connected layer
    model.add(Dense(120, activation='softmax'))
    if status:
        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), F1Score()])
        print("VGG19 Model ready for training")
        print(model.summary())
        return model

def resnet50(status,image_height,img_width):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_height, img_width, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))  # Another fully connected layer
    model.add(Dense(120, activation='softmax'))
    if status:
        model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), F1Score()])
        print("resnet50 Model ready for training")
        print(model.summary())
        return model

def efficientnet(status,image_height,img_width):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height,img_width,3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))  # Another fully connected layer
    model.add(Dense(120, activation='softmax'))
    if status:
        model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy', Precision(), Recall(), F1Score()])
        print("efficientnet Model ready for training")
        print(model.summary())
        return model


def inception(status,image_height,img_width):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(image_height,img_width,3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))  # Another fully connected layer
    model.add(Dense(120, activation='softmax'))
    if status:
        model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy', Precision(), Recall(), F1Score()])
        print("inceptionResnet Model ready for training")
        print(model.summary())
        return model

def model_fit(model,x_train,x_test,y_train,y_test):
    str = input("Do you want to train the model(yes/no):")
    if str=='yes':
        early_stopping = EarlyStopping(monitor='val_accuracy',patience=10,mode='max',restore_best_weights=True)
        print("Model training in progress")
        history=model.fit(x_train, y_train, batch_size=64, shuffle=True, epochs=100, validation_data=(x_test, y_test),callbacks=[early_stopping])
        model_evaluation = model.evaluate(x_test, y_test)
        print(f'Model Training is Done✅', '\n', f'model evaluation:-{model_evaluation}')
        return history
    else:
        return


def model_fit_with_generated_data(model, train_data_gen, x_test, y_test):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    print("Model training in progress")
    history = model.fit(train_data_gen, batch_size=4, shuffle=True, epochs=100, validation_data=(x_test, y_test),callbacks=[early_stopping])
    model_evaluation = model.evaluate(x_test, y_test)
    print(f'Model Training is Done✅', '\n', f'model evaluation:-{model_evaluation}')
    return history

def savemodel(model,x_train,y_train,x_test,y_test,file_name):
    #status = input('Do you want to save model to file? (yes/no)')
    #if status == 'yes':
    model.save(f"outputs/onstandforddataset/{file_name}.h5")
    print('Model Saved Successfully✅')
    #elif status == 'no':
    #return
    #else:
        #print('option choosen not found❌')
        #return

def plot_model_performance(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    #f1score
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.title('model f1 score')
    plt.ylabel('f1score')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def predict(img_path,model_name):
    index_to_label = mapping_labels_to_index('/Users/bhavithsmacbook/Downloads/dogbreedprediction_dataset/train')
    print(index_to_label)
    img = cv2.imread(img_path,)
    resized_img = cv2.resize(img,(224,224))
    normalized_img = resized_img/255.0
    img = np.expand_dims(normalized_img, axis=0)
    model = load_model(f"{model_name}.h5")
    prediction_probabilities = model.predict(img)[0]
    top_3 = heapq.nlargest(3, enumerate(prediction_probabilities), key=lambda x: x[1])
    for class_index, confidence in top_3:
        print(f"Class: {index_to_label.get(class_index)}, Confidence: {confidence*100:.2f}")

def mapping_labels_to_index(dataset_path):
    index_to_label = {}
    for index, label in enumerate(os.listdir(dataset_path)):
        index_to_label[index] = label
    return index_to_label


def realtime_prediction(model_path):
    breed_names = ['Afghan', 'African Wild Dog', 'Airedale', 'American Hairless', 'American Spaniel', 'Basenji',
                   'Basset', 'Beagle', 'Bearded Collie', 'Bermaise', 'Bichon Frise', 'Blenheim', 'Bloodhound',
                   'Bluetick', 'Border Collie', 'Borzoi', 'Boston Terrier', 'Boxer', 'Bull Mastiff', 'Bull Terrier',
                   'Bulldog', 'Cairn', 'Chihuahua', 'Chinese Crested', 'Chow', 'Clumber', 'Cockapoo', 'Cocker',
                   'Collie', 'Corgi', 'Coyote', 'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound',
                   'French Bulldog', 'German Shepherd', 'Golden Retriever', 'Great Dane', 'Great Pyrenees',
                   'Greyhound', 'Groenendael', 'Irish Spaniel', 'Irish Wolfhound', 'Japanese Spaniel', 'Komondor',
                   'Labradoodle', 'Labrador', 'Lhasa', 'Malinois', 'Maltese', 'Mex Hairless', 'Newfoundland',
                   'Pekinese', 'Pit Bull', 'Pomeranian', 'Poodle', 'Pug', 'Rhodesian', 'Rottweiler', 'Saint Bernard',
                   'Schnauzer', 'Scotch Terrier', 'Shar_Pei', 'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla',
                   'Yorkie']
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    def preprocess_frame(frame):
        frame_resized = cv2.resize(frame, (224, 224))  # Resize to (224x224) suitable for Inception ResNet V2
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_array = np.expand_dims(np.array(frame_rgb), axis=0)  # Add batch dimension
        return preprocess_input(frame_array)  # Preprocess for Inception-ResNet V2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Indices of the top 3 predictions
            top_3_confidences = prediction[0][top_3_indices]  # Confidence scores for the top 3
            top_3_breeds = [breed_names[i] for i in top_3_indices]  # Corresponding breed names

            for i, (breed, confidence) in enumerate(zip(top_3_breeds, top_3_confidences)):
                text = f'{i + 1}: {breed} ({confidence * 100:.2f}%)'
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam - Dog Breed Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def evaluate_model(model_path,x_test,y_test):
    model = load_model(model_path)
    val=model.evaluate(x_test,y_test)


def plot_performance_saved_models(model_path, x_test, y_test):
    # Load the saved model
    model = load_model(model_path)

    # Evaluate the model on test data
    results = model.evaluate(x_test, y_test, verbose=1)
    print(f"Loss: {results[0]}, Accuracy: {results[1]}")

    # Generate predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
    y_true = np.argmax(y_test, axis=1)  # Convert one-hot encoding to class indices

    # Calculate metrics
    precision = precision_score(y_true, y_pred_classes, average="weighted")
    recall = recall_score(y_true, y_pred_classes, average="weighted")
    f1 = f1_score(y_true, y_pred_classes, average="weighted")

    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

    # Plot Accuracy and Loss if history is available
    try:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    except:
        print("No training history available for plotting.")

    # Plotting Precision, Recall, and F1-Score
    metrics = {'Precision': precision, 'Recall': recall, 'F1-score': f1}
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Scores are between 0 and 1
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Class-wise Accuracy
    class_acc = np.array([np.sum((y_pred_classes == i) & (y_true == i)) / np.sum(y_true == i) for i in np.unique(y_true)])
    plt.bar(np.unique(y_true), class_acc)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.show()


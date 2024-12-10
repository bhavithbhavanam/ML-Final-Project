#learnings
#working with imbalanced datasets
import os
from DataPreprocessing import eda, datapreprocessing, datapreprocessing_standford_dataset, labels, imagedatageneration
from Models import simplecnn, model_fit, vgg16, savemodel, plot_model_performance, predict, vgg19, efficientnet, \
    resnet50, model_fit_with_generated_data, realtime_prediction, inception,evaluate_model,plot_performance_saved_models

def model_caller(model_name,type_data,status,x_train, x_test, y_train, y_test):
    model_name = model_name.lower()
    type_data = type_data.lower()
    if type_data=='efd':
        if model_name == 'simplecnn':
            model = simplecnn(status, 224, 224)
        elif model_name == 'vgg16':
            model = vgg16(status,224,224)
        elif model_name == 'vgg19':
            model = vgg19(status,224,224)
        elif model_name == 'efficientnet':
            model = efficientnet(status, 224, 224)
        elif model_name == 'resnet':
            model = resnet50(status,224,224) 
        elif model_name == 'inception':
            model = inception(status,299,299)
        else:
            #simpleCNN Model Training
            model = simplecnn(status, 224, 224)
            history=model_fit(model,x_train,x_test,y_train,y_test)
            file_name = 'simplecnn'
            #plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test,file_name)
            #VGG16 Training
            model = vgg16(status, 224, 224)
            history = model_fit(model, x_train, x_test, y_train, y_test)
            file_name = 'vgg16'
            #plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #vgg19
            model = vgg19(status, 224, 224)
            history = model_fit(model, x_train, x_test, y_train, y_test)
            file_name = 'vgg19'
            #plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #resnet
            model = resnet50(status, 224, 224)
            history = model_fit(model, x_train, x_test, y_train, y_test)
            file_name = 'resnet'
            #plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #efficientnet
            model = efficientnet(status, 224, 224)
            history = model_fit(model, x_train, x_test, y_train, y_test)
            file_name = 'efficientnet'
            #plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            return
        history = model_fit(model, x_train, x_test, y_train, y_test)
        file_name = model_name
        #plot_model_performance(history)
        savemodel(model, x_train, x_test, y_train, y_test, file_name)

    elif type_data=='afd':
        augumented_data = imagedatageneration(x_train,y_train,32)
        if model_name == 'simplecnn':
            model = simplecnn(status, 224, 224)
        elif model_name == 'vgg16':
            model = vgg16(status,224,224)
        elif model_name == 'vgg19':
            model = vgg19(status,224,224)
        elif model_name == 'efficientnet':
            model = efficientnet(status, 224, 224)
        elif model_name == 'resnet':
            model = resnet50(status,224,224)
        else:
            #simpleCNN Model Training
            model = simplecnn(status, 224, 224)
            history=model_fit_with_generated_data(model,augumented_data,x_test,y_test)
            file_name = 'simplecnn'
            plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test,file_name)
            #VGG16 Training
            model = vgg16(status, 224, 224)
            history=model_fit_with_generated_data(model,augumented_data,x_test,y_test)
            file_name = 'vgg16'
            plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #vgg19
            model = vgg19(status, 224, 224)
            history=model_fit_with_generated_data(model,augumented_data,x_test,y_test)
            file_name = 'vgg19'
            plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #resnet
            model = resnet50(status, 224, 224)
            history=model_fit_with_generated_data(model,augumented_data,x_test,y_test)
            file_name = 'resnet'
            plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            #efficientnet
            model = efficientnet(status, 224, 224)
            history=model_fit_with_generated_data(model,augumented_data,x_test,y_test)
            file_name = 'efficientnet'
            plot_model_performance(history)
            savemodel(model, x_train, x_test, y_train, y_test, file_name)
            return
        history = model_fit_with_generated_data(model,augumented_data, x_test, y_test)
        file_name = model_name
        plot_model_performance(history)
        savemodel(model, x_train, x_test, y_train, y_test, file_name)
    else:
        print("Selection not found")
        return

if __name__=='__main__':
    #dataset_path = r"C:\Users\swarn\Documents\Projects\DeepLearning_DataSets\StandfordDogBreedDataset\images"
    dataset_path='/Users/bhavithsmacbook/Downloads/dogbreedprediction_dataset'
    eda(dataset_path)
    model_to_be_called=input("Please enter the name of the model you want to train or use for predictions\n1.VGG16\t2.VGG19\t3.resnet\t4.SimpleCNN\n")
    type_data = input("Please enter the type of data you want to train the model with\n1.extracted features data(efd)\n")
    if model_to_be_called == 'nasnet':
        x_train, x_test, y_train, y_test, status = datapreprocessing_standford_dataset(dataset_path, 256, 256)
        x_test_predict = x_test
        y_test_predict = y_test
        model_caller(model_to_be_called, type_data, status, x_train, x_test, y_train, y_test)
        evaluate_model(f'outputs/{model_to_be_called}.h5', x_test_predict, y_test_predict)
        plot_performance_saved_models(f'outputs/{model_to_be_called}.h5', x_test_predict, y_test_predict)

    elif model_to_be_called == 'inception' or 'efficientnet' and model_to_be_called != 'vgg16' and model_to_be_called != 'vgg19' and model_to_be_called != 'resnet' and model_to_be_called != 'simplecnn':
        x_train, x_test, y_train, y_test, status = datapreprocessing_standford_dataset(dataset_path, 299, 299)
        x_test_predict = x_test
        y_test_predict = y_test
        model_caller(model_to_be_called, type_data, status, x_train, x_test, y_train, y_test)
        evaluate_model(f'outputs/{model_to_be_called}.h5', x_test_predict, y_test_predict)
        plot_performance_saved_models(f'outputs/{model_to_be_called}.h5', x_test_predict, y_test_predict)

        #model_caller(model_to_be_called, type_data, status, x_train, x_test, y_train, y_test)
    else:
        x_train, x_test, y_train, y_test, status = datapreprocessing(dataset_path)
        x_test_predict = x_test
        y_test_predict = y_test
        model_caller(model_to_be_called, type_data, status, x_train, x_test, y_train, y_test)
        evaluate_model(f'{model_to_be_called}.h5', x_test_predict, y_test_predict)
        plot_performance_saved_models(f'{model_to_be_called}.h5', x_test_predict, y_test_predict)

    pord = input("Do you want to predict in realtime(yes/no):\n")
    if pord.lower()=='no':
        img = input(r"please enter the image path you want to predict:")
        predict(img,model_to_be_called)
    else:
        realtime_prediction('vgg16.h5')
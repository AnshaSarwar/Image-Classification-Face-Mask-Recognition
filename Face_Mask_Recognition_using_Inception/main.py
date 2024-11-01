from dataset import Dataset
from trainer import Trainer
from tester import Tester


if __name__ == '__main__':


    # Load and preprocess the dataset (augment Class 2 once and cache it)
    dataset = Dataset()

    # To augment and cache Class 2 data (run once)
    # dataset.load_and_preprocess_data(augment_class_2=True, augment_factor=2)

    # Load data (use cached augmentation for Class 2)
    X_train, X_test, y_train, y_test = dataset.load_and_preprocess_data(augment_class_2=True, augment_factor=3)

    # Print the shapes of the train and test sets
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Get and print the new class distribution
    distribution = dataset.get_class_distribution(y_train)
    print("Class distribution after augmentation:", distribution)

    # Initialize Trainer and build the model with InceptionV3 and grayscale input
    #trainer = Trainer(X_train, y_train, X_test, y_test)
    #trainer.build_model(lr=0.000001, dropout_rate=0.1, trainable_layers=170)

    # Train the model
    #history = trainer.train_model(epochs=1, batch_size=64)

    # Plot accuracy and loss
    #trainer.plot_accuracy(history)
    #trainer.plot_loss(history)

    # Save the trained model
    model_path = "model_inceptionv3.h5"
    tester = Tester(X_test=X_test, y_test=y_test)
    tester.load_model(model_path)

    # Evaluate the model
    tester.evaluate_model()
    tester.evaluate_through_metrics()
    tester.predict_on_test_data()

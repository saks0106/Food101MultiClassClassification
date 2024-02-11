
# Multiclass Classification for 101 Food Categories

This project focuses on multiclass classification for a diverse set of 101 food categories. The goal is to develop a robust machine learning model that can accurately classify images into one of the predefined food categories. This README file provides essential information to understand and contribute to the project.


#### The Following are the food categories:


    1. apple_pie
    2. baby_back_ribs
    3. baklava
    4. beef_carpaccio
    5. beef_tartare
    6. beet_salad
    7. beignets
    8. bibimbap
    9. bread_pudding
    10. breakfast_burrito
    11. bruschetta
    12. caesar_salad
    13. cannoli
    14. caprese_salad
    15. carrot_cake
    16. ceviche
    17. cheese_plate
    18. cheesecake
    19. chicken_curry
    20. chicken_quesadilla
    21. chicken_wings
    22. chocolate_cake
    23. chocolate_mousse
    24. churros
    25. clam_chowder
    26. club_sandwich
    27. crab_cakes
    28. creme_brulee
    29. croque_madame
    30. cup_cakes
    31. deviled_eggs
    32. donuts
    33. dumplings
    34. edamame
    35. eggs_benedict
    36. escargots
    37. falafel
    38. filet_mignon
    39. fish_and_chips
    40. foie_gras
    41. french_fries
    42. french_onion_soup
    43. french_toast
    44. fried_calamari
    45. fried_rice
    46. frozen_yogurt
    47. garlic_bread
    48. gnocchi
    49. greek_salad
    50. grilled_cheese_sandwich
    51. grilled_salmon
    52. guacamole
    53. gyoza
    54. hamburger
    55. hot_and_sour_soup
    56. hot_dog
    57. huevos_rancheros
    58. hummus
    59. ice_cream
    60. lasagna
    61. lobster_bisque
    62. lobster_roll_sandwich
    63. macaroni_and_cheese
    64. macarons
    65. miso_soup
    66. mussels
    67. nachos
    68. omelette
    69. onion_rings
    70. oysters
    71. pad_thai
    72. paella
    73. pancakes
    74. panna_cotta
    75. peking_duck
    76. pho
    77. pizza
    78. pork_chop
    79. poutine
    80. prime_rib
    81. pulled_pork_sandwich
    82. ramen
    83. ravioli
    84. red_velvet_cake
    85. risotto
    86. samosa
    87. sashimi
    88. scallops
    89. seaweed_salad
    90. shrimp_and_grits
    91. spaghetti_bolognese
    92. spaghetti_carbonara
    93. spring_rolls
    94. steak
    95. strawberry_shortcake
    96. sushi
    97. tacos
    98. takoyaki
    99. tiramisu
    100. tuna_tartare
    101. waffles


Hungry Right! Try the Streamlit App: [Food101](https://choosealicense.com/licenses/mit/)

## Project Description:


### Dependencies 
    pip install -r requirements.txt

Ensure you have the following dependencies installed:

    Python 3. +
    TensorFlow
    Keras
    NumPy
    Matplotlib
    Pandas
    Seaborn


## Typical Workflow: 

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/Frame%204.png)




### Grabbing the Dataset: 

The dataset comprises a collection of high and low resolution images representing various food items. Each image is labeled with one of the 101 food categories. The dataset is split into training, validation, and test sets to facilitate model training, tuning, and evaluation. Dataset from https://www.tensorflow.org/datasets/catalog/food101


### Preprocessing Steps:

* Using the [`tf.keras.preprocessing.image_dataset_from_directory`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory) and passing the training dataset.

* [`Conv2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) layers (and the parameters which come with them)

* [`MaxPool2D`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) layers (and their parameters).

* The `steps_per_epoch` and `validation_steps` parameters in the `fit()` function
    steps_per_epoch - this is the number of batches a model will go through per epoch, in our case, we want our model to go through all batches so it's equal to the length of train_data (1500 images in batches of 32 = 1500/32 = ~47 steps)
    validation_steps - same as above, except for the validation_data parameter (500 test images in batches of 32 = 500/32 = ~16 steps)



#### Discussing some of the components of the Conv2D layer:
    1. The "2D" means our inputs are two dimensional (height and width), even though they have 3 colour channels, the convolutions are run on each channel invididually.

    2. filters - these are the number of "feature extractors" that will be moving over our images.

    3. kernel_size - the size of our filters, for example, a kernel_size of (3, 3) (or just 3) will mean each filter will have the size 3x3, meaning it will look at a space of 3x3 pixels each time. The smaller the kernel, the more fine-grained features it will extract.

    4. stride - the number of pixels a filter will move across as it covers the image. A stride of 1 means the filter moves across each pixel 1 by 1. A stride of 2 means it moves 2 pixels at a time.

    5. padding - this can be either 'same' or 'valid', 'same' adds zeros the to outside of the image so the resulting output of the convolutional layer is the same as the input, where as 'valid' (default) cuts off excess pixels where the filter doesn't fit



## Setting up callbacks - 

[Callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) are extra functionality we can add to your models to be performed during or after training. Some of the most popular callbacks include:
* [**Experiment tracking with TensorBoard**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) - log the performance of multiple models and then view and compare these models in a visual way on [TensorBoard](https://www.tensorflow.org/tensorboard) (a dashboard for inspecting neural network parameters). Helpful to compare the results of different models on your data.
* [**Model checkpointing**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) - save your model as it trains so you can stop training if needed and come back to continue off where you left.(Every 10 or 100 or 1000 epochs-give me the result)Helpful if training takes a long time and can't be done in one sitting.
* [**Early stopping**](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) - leave your model training for an arbitrary amount of time and have it stop training automatically when it ceases to improve. Helpful when you've got a large dataset and don't know how long training will take.


## Data augmentation: 
A preprocessing step for images when data is low or when we want to create multiple variations of same/similar image. Typically it is the process of altering our **training data only**, leading to it having more diversity and in turn allowing our models to learn more generalizable patterns. Altering might mean adjusting the rotation of an image, flipping it, cropping it or something similar.
Using data augmentation gives us another way to prevent overfitting and in turn make our model more generalizable.

The data augmentation transformations we're going to use are:
* [RandomFlip](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomFlip) - flips image on horizontal or vertical axis.
* [RandomRotation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomRotation) - randomly rotates image by a specified amount.
* [RandomZoom](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomZoom) - randomly zooms into an image by specified amount.
* [RandomHeight](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomHeight) - randomly shifts image height by a specified amount.
* [RandomWidth](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomWidth) - randomly shifts image width by a specified amount.
* [Rescaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling) - normalizes the image pixel values to be between 0 and 1, this is worth mentioning because it is required for some image models but since we're using the `tf.keras.applications` implementation of `EfficientNetB0`, it's not required.



# Steps in modelling

Now we know what data we have as well as the input and output shapes as well as above mentioned preprocessing steps, data augmentation and callbacks , We will see how we'd build a convolutional neural network **Model**

In TensorFlow, there are typically 3 fundamental steps to creating and training a model.

1. **Creating a model** - piece together the layers of a neural network (using the [functional](https://www.tensorflow.org/guide/keras/functional) or [sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)) or import a previously built model (known as transfer learning).

2. **Compiling a model** - defining how a model's performance should be measured (loss/metrics) as well as defining how it should improve (optimizer). 

3. **Fitting a model** - letting the model try to find patterns in the data (how does `X` get to `y`). 

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/Cnn_Layers.JPG)


### Model Architecture EfficientNetB0 and ResNetV250: 
The both models have convolutional neural network (CNN) architecture and effective for image classification tasks. The model consists of multiple convolutional layers, followed by pooling layers and fully connected layers. Transfer learning with pre-trained models such as ResNet or EfficientNet is employed to boost performance.

### Building a typical model using the Keras Functional API :
Using the Keras Functional API [`tf.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications) module as it contains a series of already trained (on ImageNet) computer vision models as well as the Keras Functional API to construct our model.

Steps to be followed:

    1. Instantiate a pre-trained base model object by choosing a target model such as [`EfficientNetB0`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) from `tf.keras.applications`, setting the `include_top` parameter to `False` (we do this because we're going to create our own top, which are the output layers for the model).
    2. Set the base model's `trainable` attribute to `False` to freeze all of the weights in the pre-trained model.
    3. Define an input layer for our model, for example, what shape of data should our model expect?
    4. Normalize the inputs to our model if it requires.
    5. Pass the inputs to the base model.
    6. Pool the outputs of the base model into a shape compatible with the output activation layer (turn base model output tensors into same shape as label tensors). This can be done using [`tf.keras.layers.GlobalAveragePooling2D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D) or [`tf.keras.layers.GlobalMaxPooling2D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool2D?hl=en) though the former is more common in practice.
    7. Create an output activation layer using `tf.keras.layers.Dense()` with the appropriate activation function and number of neurons.
    8. Combine the inputs and outputs layer into a model using [`tf.keras.Model()`](https://www.tensorflow.org/api_docs/python/tf/keras/Model).
    9. Compile the model using the appropriate loss function and choose of optimizer.
    10. Fit the model for desired number of epochs and with necessary callbacks (in our case, we'll start off with the TensorBoard callback).


### Model Evaluation method:
    Accuracy: 	Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct. 
    Library: 	sklearn.metrics.accuracy_score() or tf.keras.metrics.Accuracy()

    Precision: 	Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0).	
    Library: sklearn.metrics.precision_score() or tf.keras.metrics.Precision()

    Recall: Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives.	
    Library: sklearn.metrics.recall_score() or tf.keras.metrics.Recall()

    F1-score:	Combines precision and recall into one metric. 1 is best, 0 is worst.	
    Library: sklearn.metrics.f1_score()

    Classification report: 	Collection of some of the main classification metrics such as precision, recall and f1-score.	
    Library: sklearn.metrics.classification_report()

    Confusion matrix:	Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).	
    Library: Custom function or sklearn.metrics.plot_confusion_matrix()

#### A simple binary Confusion Matrix: 
![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/confusion_matrix.jpg)

#### Our Food101 Multiclass Confusion Matrix:
![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/model_cm.png)



### Plot the history and loss curves:
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))


![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/model_loss.png)

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/model_accuracy.png)


### Note: 
    As the dataset contains 250 images per 101 classes which adds up to 25250 images for training dataset and 75 images per 101 classes which adds up to 7575 images for validation and testing dataset, the GPU and RAM were limited.
    
    Model could have been improved from ~60% F1 score to atleast ~80% F1 Score if Google Colab Paid Service was used!



## Hyperparameters  Tuning:

1. **The activation parameter** - We used strings ("relu" & "sigmoid") instead of using library paths (tf.keras.activations.relu), in TensorFlow, they both offer the same functionality.
2. **The learning_rate parameter** - We increased the learning rate parameter in the Adam optimizer to 0.01 instead of 0.001 (an increase of 10x).
3. **The number of epochs** - We lowered the number of epochs (using the epochs parameter) from 100 to 25 but our model still got an incredible result on both the training and test sets.

4. **The optimizers** - By default Adam optimier i.e default settings can usually get good results on many datasets


## Transfer Learning Techniques: 

1. **"Transfer Learning** is when we take a pretrained model as it is and apply it to your task without any changes.

2. **Feature extraction transfer learning** is when we take the underlying patterns (also called weights) a pretrained model has learned and adjust its outputs to be more suited to your problem. We just remove the original activation layer and replace it with your own but with the right number of output classes. The important part here is that **only the top few layers become trainable, the rest remain frozen**

3. **Fine-tuning transfer learning** is when we take the underlying patterns of a pretrained model and adjust (fine-tune) them to our own problem. This usually means training **some, many or all** of the layers in the pretrained model. This is useful when you've got a large dataset


After we have trained the top n layers, you can then gradually "unfreeze" more and more layers and run the training process on our own data to further **fine-tune** the pretrained model. We also reduce the learning_rate so that image outlines and patterns are well learned by the CNN model

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/06-lowering-the-learning-rate.png)

### Post Training Results:

Model was trained for the first 5 epochs before saving the model and Fine-Tuning the saved model with EfficientNetB0 pre-trained weights. The Results are Below:

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/Loss%20and%20Accracy_model.JPG)

When it comes to Classification Machine Learning Problem,F1 is looked as an important metric to measure the effectiveness of the trained model. Below are the F1 scores for 101 classes 

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/f1_model.png)

We have found Correct Predictions but it is equally important to find the wrong predictions i.e why the model is predicting a wrong class so **confidently**. 
Few Reasons could be:

    1. Wrong Image Labels
    2. Similar image in color or texture like Pancakes and Waffles
    3. Custom Image parameters not seen by the model i.e model has not been trained on that Custom Image 

![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/Model_Wrong_Preds.JPG)



Demo of our wrong predictions:
![](https://raw.githubusercontent.com/saks0106/Food101MultiClassClassification/main/Explaination_ScreenShots/Wrong_Preds.png)




## CONTRIBUTIONS

Any contributions are welcomed! If you have suggestions, bug reports, or want to add new features or any queries, please don't hesitate to contact me.























    
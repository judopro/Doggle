# Doggle

Dog Breed Classification using Machine Learning - Article link : 
https://medium.com/@judopro/image-recognition-transfer-learning-great-performance-with-limited-data-keras-tensorflow-1299c87f8423

Download data set from http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

Use <b>create_validation_set.sh</b> script to create a separate validaton set randomly from within the images and move them to a separate validation folder. Just run it wuthout any parameters, make sure current images are extracted into "train" folder AND you are running this script from withint "train" folder.

Once validation data set is created. You can start the model to train and then predict...

<b>python doggle_cnn_v8.py --mode train</b>

<b>python doggle_cnn_v8.py --mode predict --image PATH_TO_YOUR_IMAGE.JPG</b>

Your folder structure should be 

./doggle_cnn_v8.py
/train/TRAINING_IMAGES <- Run create_validation_set.sh here after download image dataset and expanding into this folder
/validation/VALIDATION_IMAGES

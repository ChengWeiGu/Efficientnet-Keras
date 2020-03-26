## The efficientnet model is run under keras, please refer to the following steps:

## Model Installation:
Please refer to the following webstite:  
https://github.com/qubvel/efficientnet

my envs:   
tensorflow-gpu1.8.0  
keras2.2.0  
cudnn7.6.5

## Run data Preprocess:
Please use the script "Train_Test_Prep_forColor.py" to pickle the image data in dataset.  
It will generate three files including "filenames.csv", "datasets.pkl" and "datasets_split.pkl".


## Run the model:
Please run the script "EfficientnetB5.py" after finishing data preprocess.  
Note that the "datasets_split.pkl" is needed

command: python EfficientnetB5 -t

## Reference:
The script is revised to fit my requirement, the original script can be sourced to:  
https://www.kaggle.com/ateplyuk/mnist-efficientnet/  

You can paraphrase it if needed!


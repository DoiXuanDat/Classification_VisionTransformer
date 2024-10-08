# Classification with Vision Transformer 

Paper: https://arxiv.org/abs/2010.11929

## Dataset

This project using Garbage classification dataset from 
kaggle https://www.kaggle.com/datasets/yashkangale20/garbage-classification

The structure of dataset was shown as follows:

- data
- ├── Class_1
- │   ├── img_1-1.jpg
- │   ├── img_1-2.jpg
- │   └── ...
- ├── Class_2
- │   ├── img_2-1.jpg
- │   ├── img_2-2.jpg
- │   └── ...
- ├── Class_3
- │   ├── img_3-1.jpg
- │   ├── img_3-2.jpg
- │   └── ...
- └── ...
  
## Hyperparameters

- Batch Size = 1280
- Epochs = 50
- Learning Rate = 1e-4
- Patch Size = 16
- Hidden Size = 64
- Number of Hidden Layers = 3
- Number of Attention Heads = 5
- Intermediate Size = 256
- Hidden Dropout = 0.09
- Attention dropout  = 0.02
  
## Results

### Evaluation Metrics: 

- Accuracy: Achieved an overall accuracy of 80.46% on the test dataset
- Loss: The model's performance is hindered by a high test loss of 0.4943

The model have medium performance on small pack dataset 








## Slice Registration

**SliceAutoMapi**

>High-throughput Automated Mapping of 2D Mouse Brain Image Slices to 3D Brain Volumes

✨ ResNet+Transformer

📖 Introduction

SliceAutoMapi: Prepared for SliceAutoMapi: High-throughput Automated Mapping of 2D Mouse Brain Image Slices to 3D Brain Volumes

📦 Prerequisites

- OS: Windows
- Runtime: Python3
- Tools: Git

🚀 Quick Start

1.Clone the Repository

```bash
git clone https://github.com/SlicesMapi/
cd YourRepo ./SliceAutoMapi
```

2.Install Dependencies

The python version we used is 3.6.5. tensorflow=1.15. Newer Python releases do not support TensorFlow 1.15.
We recommend using Anaconda to set up a dedicated virtual environment. 
And we are working on reproducing the results under newer Python and TensorFlow environments.
The code supporting more versions will be uploaded within the next few weeks.

Anaconda official download address:
https://www.anaconda.com/download

```bash
conda create -n SliceAutoMapi python=3.6.5
conda activate SliceAutoMapi
```

go to your path of sliceautomapi

```
cd YourRepo ./SliceAutoMapi
pip install -r requirements.txt
```

3.Run the Project

```bash
# Train
python train_model.py
# Predict Mode1
python predict_SliceAutoMapi.py config_dir data_dir save_dir 0 slice_num
# Predict Mode2
python predict_SliceAutoMapi.py config_dir data_dir save_dir interval slice_num
```

💡 Usage Example

```bash
python predict_SliceAutoMapi.py ./Config_new.ini ./image_test/ ./image_predict/ 10 50
```

📦 Model Weights

Pretrained model weights for test are available on Hugging Face, more model weights for different usage will upload soon. (https://huggingface.co/ztzhang123/SliceAutoMapi/tree/main). 
Please download 'model'  'reference' 'traindata' to the main directory of sliceautomapi if you want to test the model.

The download 'model'  'reference' 'traindata' are used in Config.ini when you run the code.

```text
[DIR]
#traindatasets
train_dir=./traindata/
#model   
model_dir=./model/1/
#reference data           
fixed_dir=./reference/atlas.tif
annotation data
ann_dir=./reference/atlas.tif
[Default]
mk=2
n_iter=100
subject_id='fixed_mask2_25'
debug=store_true
```

📂 Project Structure

```text
SliceAutoMapi
├── generate_data/        # Code for generate trainsets
├── geomstats/          # Code bk
├── model/           # model dir
├── traindata/           # traindata dir
├── reference/          # Reference atlas
├── antsApplyTransforms.exe    # ants
├── antsRegistration.exe    # ants
├── registration_mask.sh    # ants code for non-rigid registration
└── README.md      # Project description
```

🤝 Contributing

Welcome all developers to submit Issues and Pull Requests to improve this project together!

1. Fork this repository
2. Create your feature branch: git checkout -b feat/xxx
3. Commit your changes: git commit -m 'feat: add new feature'
4. Push to the branch: git push origin feat/xxx
5. Submit a Pull Request

🐛 Issues & Feedback

If you encounter bugs or have feature suggestions during use, please submit them via Issues. We will reply and fix them as soon as possible.

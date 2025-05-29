
This includes the scripts and code for preprocessing steps and relevant code to obtain the training motion data from the raw data.

You need to perform the following steps:


```
cd ./data_generation/data_preprocess
git clone https://github.com/EricGuo5513/HumanML3D.git
mv ./data_generation/data_preprocess/motion_representation_AniMo4D.ipynb ./data_generation/data_preprocess/HumanML3D/
cp ./utils/paramUtil.py ./data_generation/data_preprocess/HumanML3D/
```

Then, run the `motion_representation_AniMo4D.ipynb` notebook to generate the preprocessed motion data.

This directory contains the code and scripts used to generate the AniMo4D dataset.

Please follow the instructions provided in the `export_json`, `json2npy`, and `data_preprocess` folders' respective `README.md` files. By doing so, you will be able to generate the full set of motions included in the AniMo4D dataset.

For the textual descriptions and dataset splits, you can obtain them from the following link: [AniMo4D](https://1drv.ms/f/c/079F683083C0B60F/Erfn6rrOGU5KmRSU85AlLZ8BiuNpS9-G7ucmOsHMEYP8YQ?e=k31xgW)

After preprocessing, move all data into the `AniMo4D` folder.


📌 Note on data length:

The original data sequences ranged from 20 to 300 frames as described in the paper. However, due to a 1-frame reduction during preprocessing, the actual data used in the released version ranges from 19 to 299 frames.

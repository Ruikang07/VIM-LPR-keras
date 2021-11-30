# VIM-LPR-keras
This is a re-implementation of VIM-LPR for paper "Efficient License Plate Recognition via Holistic Position Attention"(AAAI2021) using keras.
<br>
This code is based on jfzhuang's [VIM-LPR](https://github.com/jfzhuang/VIM-LPR)
<br><br>

# Conda Environment
python                  3.7.11

tensorflow              2.6.0

scikit-learn             1.0.1

opencv                    4.5.1

matplotlib               3.4.0


# Data preparation

You need to download the [AOLP](http://aolpr.ntust.edu.tw/lab/) datasets.<br>

The plates char need to be converted to pure number format<br>

Put all no-skewed plates under  data/AOLP_noSkew<br>
AOLP_noSkew = AOLP_AC + AOLP_LE + a few noSkewed plates from AOLP_RP<br>

or<br>

All the image files and label files can be saved in npy files<br>
<pre>
├── data
│     ├── AOLP_noSkew
│     │     ├── label_in_number
│     │     └── image
│     │   
│     ├── AOLP_AC_plate_images.npy  
│     ├── AOLP_AC_plate_labels_in_number.npy
│     ├── AOLP_LE_plate_images.npy  
│     ├── AOLP_LE_plate_labels_in_number.npy  
│     ├── AOLP_noSkew_plate_images.npy  
│     └── AOLP_noSkew_plate_labels_in_number.npy


</pre>

# For quick predicting:
1. download the pretrained model weights file: [lpr_model_weights.h5](https://drive.google.com/file/d/1tGOftwEzOXETH8k4qvEGdRA0Tuir2KdP/view?usp=sharing). It was trained on 
   AOLP_noSkew dataset
2. put it under pretrained_model
3. specify the plate image file in lpr_predict.py
4. run: python lpr_predict.py


# For training:
1. prepare the AOLP dataset
2. run: python lpr_train.py 
<br>(support npy data format only at present)


# Accuracy of pretrained model on AOLP AC and LE:
1. Accuracy on AOLP AC is 93.51%
2. Accuracy on AOLP LE is 96.55% 


# Citation

@inproceedings{zhang2021efficient,
  title={Efficient License Plate Recognition via Holistic Position Attention},
  author={Zhang, Yesheng and Wang, Zilei and Zhuang, Jiafan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{AOLP2013,
  title={Application-Oriented License Plate Recognition},
  author={Hsu, G.S.; Chen, J.C.; Chung, Y.Z.},
  booktitle={Vehicular Technology, IEEE Transactions on , vol.62, no.2},
  pages={552-561},
  year={2013}
}



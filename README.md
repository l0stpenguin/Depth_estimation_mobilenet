# Depth Estimation with Transfer Learning pretrained MobileNetV2


This project implements a deep learning neural network model to generate the depthe image of a given image.
Model is U-net model with MobileNetV2 as the encoder, and model has utilized skip connection from encoder to decoder.
Model generates a depth image of resolution 480x640 for the same sized input image.


## Getting Started

Model can be trained using the "train_mobilenetv2.ipynb".
 
IPYTHON file "test_img.ipynb" can be used to generate the depth image on pretrained model.

IPYTHON file "test_video.ipynb" can be used to generate the depth video on pretrained model.
For pretrained models see below.  

##Data 
* [NYU Depth V2 (50K)](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip) (4.1 GB): File is extraced while running the "train_mobilenetv2.ipynb".

## Download a pretrained model 
* [Mobilenet](https://drive.google.com/drive/folders/1rDvtiwUgYbhzk8ZPdQ176abv-u6SaZzI?usp=sharing) (55 MB). Pretrained model is trained on 6 NVIDIA GeForce GTX 1080 for 6 hours(17 epoches). 

### Results

A sample ECG signal segment:

![ECG image](https://github.com/alinstein/Depth_estimation/blob/master/CombineGIF.gif)

Final sparse coefficient vector representation of the above ECG signal:

![Sparse coefficient](https://github.com/alinstein/Human-Identification-with-ECG--/blob/master/observation/maxpol2.jpg)
Following table shows the results obtained:

![Result image](https://github.com/alinstein/Human-Identification-with-ECG--/blob/master/Results.JPG)

## References

[1] J. Wang, M. She, S. Nahavandi and A. Kouzani, "Human Identification From ECG Signals Via Sparse Representation of Local Segments," in IEEE Signal Processing Letters, vol. 20, no. 10, pp. 937-940, Oct. 2013.
doi: 10.1109/LSP.2013.2267593
[2] W.-S. Lu, Course notes of Advanced Mathematical optimizations. 

## Authors

%Written by Alinstein Jose, University of Victoria.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


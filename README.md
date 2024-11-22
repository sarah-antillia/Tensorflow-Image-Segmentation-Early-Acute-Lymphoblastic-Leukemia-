<h2>Tensorflow-Image-Segmentation-Early-Acute-Lymphoblastic-Leukemia (2024/11/22)</h2>

This is the second experiment of Image Segmentation for Acute-Lymphoblastic-Leukemia 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
 <a href="https://drive.google.com/file/d/1gaZzfv4ZtjMao0WYJ197oGVCmjD40NBt/view?usp=sharing">
Malignant-Early-Acute-Lymphoblastic-Leukemia-ImageMask-Dataset.zip (512x512 pixels)</a>, which was derived by us from  
<a href="https://www.kaggle.com/datasets/mehradaria/leukemia">Acute Lymphoblastic Leukemia (ALL) image dataset
</a><br>
<br>
On our dataset, please refer to:<a href="https://github.com/atlan-antillia/Image-Segmentation-Acute-Lymphoblastic-Leukemia">
Image-Segmentation-Acute-Lymphoblastic-Leukemia</a><br>

<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-016.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-016.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-016.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-033.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-033.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-033.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Early-Acute-Lymphoblastic-LeukemiaSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>

The image dataset used here has been taken from the following kaggle web site.
<a href="https://www.kaggle.com/datasets/mehradaria/leukemia">Acute Lymphoblastic Leukemia (ALL) image dataset
</a><br>

If you use this dataset in your research, please credit the authors. <br>

<b>Data Citation:</b><br> 
Mehrad Aria, Mustafa Ghaderzadeh, Davood Bashash, Hassan Abolghasemi, Farkhondeh Asadi, and Azamossadat Hosseini,<br>
“Acute Lymphoblastic Leukemia (ALL) image dataset.” Kaggle, (2021).<br>
 DOI: 10.34740/KAGGLE/DSV/2175623.<br>
<br>
<b>Publication Citation:</b><br> 
Ghaderzadeh, M, Aria, M, Hosseini, A, Asadi, F, Bashash, D, Abolghasemi, H. <br>
A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification <br>
using peripheral blood smear images.<br>
 Int J Intell Syst. 2022; 37: 5113- 5133. doi:10.1002/int.22753<br>
<br>
<h3>
<a id="2">
2 Early-Acute-Lymphoblastic-Leukemia ImageMask Dataset
</a>
</h3>
 If you would like to train this Early-Acute-Lymphoblastic-Leukemia Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1gaZzfv4ZtjMao0WYJ197oGVCmjD40NBt/view?usp=sharing">
Malignant-Early-Acute-Lymphoblastic-Leukemia-ImageMask-Dataset.zip (512x512 pixels)</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Early-Acute-Lymphoblastic-Leukemia
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<br>
<b>Early-Acute-Lymphoblastic-Leukemia Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/Early-Acute-Lymphoblastic-Leukemia_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_Ssample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Early-Acute-Lymphoblastic-LeukemiaTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemiaand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 55  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/train_console_output_at_epoch_55.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Early-Acute-Lymphoblastic-Leukemia.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/evaluate_console_output_at_epoch_55.png" width="720" height="auto">
<br><br>Image-Segmentation-Early-Acute-Lymphoblastic-Leukemia

<a href="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Early-Acute-Lymphoblastic-Leukemia/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.1295
dice_coef,0.8274
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Early-Acute-Lymphoblastic-Leukemia.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-003.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-058.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-058.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-058.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-105.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-041.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-041.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-041.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-046.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-046.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-046.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-085.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-085.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Early-Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-085.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Acute Lymphoblastic Leukemia (ALL) image dataset. Kaggle, (2021)</b><br>
Mehrad Aria, Mustafa Ghaderzadeh, Davood Bashash, Hassan Abolghasemi, Farkhondeh Asadi, and Azamossadat Hosseini,<br>

 DOI: 10.34740/KAGGLE/DSV/2175623.<br>
<br>

<b>2.A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification using peripheral blood smear images</b>
 <br>
Ghaderzadeh, M, Aria, M, Hosseini, A, Asadi, F, Bashash, D, Abolghasemi, H. <br>
<a href="https://onlinelibrary.wiley.com/doi/full/10.1002/int.22753">https://onlinelibrary.wiley.com/doi/full/10.1002/int.22753</a>
<br>



# RED-Net

This repository is implementation of the "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections". <br />
To reduce computational cost, it adopts stride 2 for the first convolution layer and the last transposed convolution layer.

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

<table>
    <tr>
        <td><center>Input</center></td>
        <td><center>JPEG (Quality 10)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_jpeg_q10.png" height="300"></center>
    	</td>
    </tr>
    <tr>
        <td><center>AR-CNN</center></td>
        <td><center><b>RED-Net 10</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_ARCNN.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_REDNet10.png" height="300"></center>
        </td>
    </tr>
    <tr>
        <td><center><b>RED-Net 20</b></center></td>
        <td><center><b>RED-Net 30</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_REDNet20.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_REDNet30.png" height="300"></center>
        </td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python main.py --arch "REDNet30" \  # REDNet10, REDNet20, REDNet30               
               --images_dir "" \
               --outputs_dir "" \
               --jpeg_quality 10 \
               --patch_size 50 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

### Test

Output results consist of image compressed with JPEG and image with artifacts reduced.

```bash
python example --arch "REDNet30" \  # REDNet10, REDNet20, REDNet30
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \
               --jpeg_quality 10               
```

# Image-super-resolution
Reconstruct a high resolution image from low resolution image

## To reproduce the result ,please run

`python main.py --dataset HW --cuda  --upscale_factor 3 --crop_size 256 --batch_size 10 --test_batch_size 5 --epochs 30 --clip 1 --step 20 --lr 1e-2`

You can modify the batch_size to fit you GPU

Run the following and you can check the result in the folder ./submission

`python run.py  --scale_factor 3 --model model_epoch_30.pth --folder ./testing_lr_images --cuda`

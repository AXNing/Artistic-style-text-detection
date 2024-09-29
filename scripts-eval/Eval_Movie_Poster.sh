
cd ../
CUDA_LAUNCH_BLOCKING=1 python3 eval_Art_text.py --net resnet50 --scale 1 --exp_name Movie_Poster --checkepoch 370  --viz   --test_size 640 1024 --dis_threshold 0.05  --cls_threshold 0.8  --gpu "0" ;

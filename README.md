This is an  old type from:https://github.com/unsky/FPN-caffe
Because author for https://github.com/unsky/FPN-caffe has changed his type,My friends cannot operate ，so I upload
this type,which can operate currently!

Feature Pyramid Network on caffe

This is the unoffical version  Feature Pyramid Network for Feature Pyramid Networks for Object Detection https://arxiv.org/abs/1612.03144


the mxnet unoffical version  Feature Pyramid Network: https://github.com/unsky/Feature-Pyramid-Networks 
# usage
dowload VGG16 `VGG16.v2.caffemodel` and your dataset
### make caffe
```
cd FP-caffe
mkdir build
cd build
cmake ..
make -j16 all
```
### make lib

```
cd lib
make 
```
### train

./experiments/scripts/FP_Net_end2end.sh 1 VGG16 pascal_voc

### test
./test.sh 1 VGG16 pascal_voc

if you have issue about the fpn, open an issue.

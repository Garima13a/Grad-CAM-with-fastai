from fastai import *
from fastai.vision import *
from fastai.core import *
from gradcam import *

#This code is for loading pretrained model for 6 class classification 
#This code explains using gradcam for fastai
# https://nbviewer.jupyter.org/github/anhquan0412/animation-classification/blob/master/gradcam-usecase.ipynb




print("let's start")
path1 = "/data/neuroretinal/Combined"
tfms = get_transforms(do_flip=False,flip_vert=True)
print("getting databunch")
data = ImageDataBunch.from_folder(path1, ds_tfms=tfms, size=224)
data.normalize()
print("got databunch")

learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.load("/data/neuroretinal/Combined/models/biop-fovnew", strict=False, remove_module=True)
print(data.classes)

test_img = '/home/g/gv53/tmp/test.jpg'
img = open_image(test_img)

gcam = GradCam.from_one_img(learn,img)
gcam.plot()
# plt.imshow(gcam)
# plt.savefig('/home/g/gv53/tmp/cam.jpg')

print('done')



# /home/garima/Desktop/UoL_Projects/6_class_classification/dataset/Combined/train/Normal/

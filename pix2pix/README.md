This notebook will work provided the fact that not only data are present in the correct directory but also that they are in the correct shape. The network was implemented to work on the facades dataset where each jpg file consists of a pair of images: ground truth on the left and observed image on the right, both of which are of shape 256x256. If it is needed to use the network on a different dataset these parameters must be adjusted. The create_dataset notebook in the g_waves_dataset folder creates a dataset with these exact properties.
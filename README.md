# SSD_TensorFlow_v1.14

- [x] Convert raw data into tfrecord files.
- [x] Read data from tfrecord files and display image.
- [x] Draw bounding boxes.
- [x] Resize image with bounding boxes.
- [x] Get anchor boxes.
    - [ ] Anchor for single layer. ==> the ratio of h and w
- [ ] Preprocessing
    - [x] Crop
    - [x] Resize
    - [x] Permutation
    - [ ] Withen

- [x] Anchors   <br/> 
    It is a list with len equals 6, and each element is a tuple representing (y, x, h, w).

- [x] Encode anchor boxes的处理步骤:
    - [x] image_preprocessing_fn ==> 处理单张image
    - [x] ssd_net.bboxes_encode()   ==> 也是处理单张image
    - [x] tf_utils.reshape_list ==> 将bboxes_encode之后的几个list进行reshape，
    使每个image处理后都有相同的size， 以便后续的batch。
    
- [ ] Encode bboxes:
    - [ ] glabels
    - [ ] gscores
    - [ ] glocalizations

- [ ] l2 normalization: undo right now

原代码数据处理的方式是：
1. Get element from tf.data.Dataset
2. Image_preprocessing
3. Encode with ssd_anchors
4. Batch




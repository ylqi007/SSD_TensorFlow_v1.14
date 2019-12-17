# SSD_TensorFlow_v1.14

- [x] Convert raw data into tfrecord files.
- [x] Read data from tfrecord files and display image.
- [x] Draw bounding boxes.
- [x] Resize image with bounding boxes.
- [ ] Get anchor boxes.
    - [ ] Anchor for single layer. ==> the ratio of h and w
- [ ] Preprocessing
    - [x] Crop
    - [ ] Resize
    - [ ] Withen
    - [ ] Crop

- [ ] Encode anchor boxes的处理步骤:
    - [ ] image_preprocessing_fn ==> 处理单张image
    - [ ] ssd_net.bboxes_encode()   ==> 也是处理单张image
    - [ ] tf_utils.reshape_list ==> 将bboxes_encode之后的几个list进行reshape，
    使每个image处理后都有相同的size， 以便后续的batch。
    


原代码数据处理的方式是：
1. Get element from tf.data.Dataset
2. Image_preprocessing
3. Encode with ssd_anchors
4. Batch

# SSD_TensorFlow_v1.14

- [x] Convert raw data into tfrecord files.
- [x] Read data from tfrecord files and display image.
- [x] Draw bounding boxes.
- [x] Resize image with bounding boxes.
- [ ] Get anchor boxes.
    - [ ] Anchor for single layer. ==> the ratio of h and w
    

原代码数据处理的方式是：
1. Get element from tf.data.Dataset
2. Image_preprocessing
3. Encode with ssd_anchors
4. Batch

# 1. SSD anchors

> **Convolutional predictors for detection**
> * Each added feature layer (or optionally an existing feature layer from the base network) can produce a fixed set of
> detection predictions using a set of convolutional filters. [即通过a set of convolutional filters之后，每一层feature layer
> 会输出一系列的detection predictions, 这些detections包括对class的predict， 也包括对location的predict。]
> ![SSD Framework](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&ved=2ahUKEwjDjdC9vcLmAhVIip4KHVdDBiwQjRx6BAgBEAQ&url=https%3A%2F%2Ftowardsdatascience.com%2Funderstanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab&psig=AOvVaw1oRDN0pKBT3XDtqqFyWE1W&ust=1576871289302599)
> 
> * For a feature layer of size $$m\times n$$ with $$p$$ channels, the basic element for predicting parameters of a 
> potential detection is a $$3 \times 3 \times p$$ small kernel that produces either a score for a category, or a shape 
> offset relative to the default box coordinates. At each of the $$m \times n$$ locations where the kernel is applied, 
> it produces an output value. [即每个small kernel会输出一个predict parameter，这个parameter可以是predict class的score，也可以是
> predict location的offset。]
> 
> * The bounding box offset output values are measured relative to a default box position relative to each feature map 
> location.

anchor字面的意思是锚，指用于固定船的东西，anchor在computer vision中有锚点或锚框之意，目标检测中长出现的anchor box指的就是锚框，代表固定的参考框。


# 2. bboxes_encode()

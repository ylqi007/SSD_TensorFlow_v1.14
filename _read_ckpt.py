import tensorflow as tf
import os
import sys


def extracting(meta_dir):
    num_tensor = 0
    var_name = ['2-convolutional/kernel']
    model_name = meta_dir
    configfiles = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(model_name)
        for f in fnmatch.filter(files, '*.meta')] # List of META files

    with tf.Session() as sess:
        try:
            # A MetaGraph contains both a TensorFlow GraphDef
            # as well as associated metadata necessary
            # for running computation in a graph when crossing a process boundary.
            saver = tf.train.import_meta_graph(configfiles[0])
       except:
           print("Unexpected error:", sys.exc_info()[0])
       else:
           # It will get the latest check point in the directory
           saver.restore(sess, configfiles[-1].split('.')[0])  # Specific spot

           # Now, let's access and create placeholders variables and
           # create feed-dict to feed new data
           graph = tf.get_default_graph()
           inside_list = [n.name for n in graph.as_graph_def().node]

           print('Step: ', configfiles[-1])

           print('Tensor:', var_name[0] + ':0')
           w2 = graph.get_tensor_by_name(var_name[0] + ':0')
           print('Tensor shape: ', w2.get_shape())
           print('Tensor value: ', sess.run(w2))
           w2_saved = sess.run(w2)  # print out tensor
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
### Merge BN and Quantizaiton

I think doing things from scratch is nonsense. Hope this helps and saves your precious time. The basic solution is to merge BN first and then quantize.  

#### *Merge BN*

BN层在训练中可以将数据归一化并且加速训练拟合速度，但是在inference阶段，BN层占据了大量的内存和显存空间，降低整体计算速度。

BN层一般位于卷积层之后：

* 卷积: ![$X = \omega * x$](http://latex.codecogs.com/gif.latex?%24X%20%3D%20%5Comega%20*%20x%24)

* BN 分为两个阶段 ：

  （第一阶段）： $ f1 = \frac{X - mean}{\sqrt[2][var]}$

  （第二阶段）： $f2 = \beta * f1 + \gamma = \beta *  \frac{\omega * x - mean}{\sqrt[2][var]} + \gamma = \beta\frac{\omega}{\sqrt[2]{var}}x + \gamma - \beta\frac{mean}{\sqrt[2]{var}}$

  $\omega_{new} = \beta\frac{\omega}{\sqrt[2]{var}}$

  $\beta_{new} = \gamma - \beta\frac{mean}{\sqrt[2]{var}}$

##### Caffe Version

Check this blog [Merge BN and Scale](https://blog.csdn.net/diye2008/article/details/78492181). This blog provides detailed solution.

##### Tensorflow Version

 Tensorflow has its own tools to merge BN. 需要用bazel编译对应的工具，需要使用Tensorflow最新的版本。

* Tensorflow Embedded Tools

  （1）Install Bazel

  - sudo apt-get install openjdk-8-jdk
  - echo "deb [arch=amd64] <http://storage.googleapis.com/bazel-apt> stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
  - curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
  - sudo apt-get update
  - sudo apt-get install bazel

  （2）Freeze Model

  - Get ouput node name

    Check this function

    ```python
    def OutputNodeName(meta):
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph()
            graph_def = tf.get_default_graph().as_graph_def()
            node_list=[n.name for n in graph_def.node]
        return node_list[-1]
        
    # meta is your graph structure
    # meta file name is like "model.ckpt.meta"
    ```

  - Transform data from ckpt to pb

    Check function freeze_graph (You can also refer to tensorflow embedded tools to freeze graph)

    ```python
    def freeze_graph(model_dir, output_node_names):
        if not tf.gfile.Exists(model_dir):
            raise AssertionError("Export directory doesn't exists. Please specify an export directory: %s" % model_dir)
    
        if not output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1
    
        # retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
    
        # precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/frozen_model.pb"
    
        # clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True
    
        # start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    
            # restore the weights
            saver.restore(sess, input_checkpoint)
    
            # use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names.split(",") # The output node names are used to select the usefull nodes
            )
    
            # Finally serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
    
        return output_graph_def
    ```

  - Build Tensorflow tools (Graph Transform Tools)

    - excute `touch WORKSPACE` in tensorflow root path
    - excute `bazel build --package_path $(tensorflow root path)/tools/graph_transforms:transform_graph` in tensorflow root path

  - Merge BN using Graph_transform tools

    ```shell
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=/home/user/3dunet/Unet_training/nobn/test.pb --out_graph=/home/user/3dunet/Unet_training/nobn/optimized_inception_graph.pb --inputs='input/Inputdata' --outputs='save/restore_all' --transforms='
      strip_unused_nodes(type=float, shape="1,32,32,32,2")
      remove_nodes(op=Identity, op=CheckNumerics)
      fold_constants(ignore_errors=true)
      fold_batch_norms
      fold_old_batch_norms
      quantize_weights
      strip_unused_nodes
      sort_by_execution_order'
    ```

* Work for me methods

#### *Quantization int8*

##### Caffe Version

Basically, quantization should be done after the BN has been merged.

As we know, int8 math has higher throughput and lower memory requirements but with significantly lower precision and dynamic range. A blog written in youdao shows the int8 quantization based on tensorRT [[The implement of Int8 quantize base on TensorRT](https://note.youdao.com/share/?id=829ba6cabfde990e2832b048a4f492b3&type=note#/)].

Check this repo [Caffe Int8](https://github.com/lyk125/caffe-int8-convert-tools.git), directly use the tools it provide.

##### Tensorflow Version

For tensorflow quantization, you can also use the `Grap_Transform tools`




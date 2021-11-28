# MoNuSAC-Instance-Segmentation

> The following steps worked for me:
> 
> **1. Upgrade the scripts by using the following line on the root folder:**
> 
> ` tf_upgrade_v2 --intree Mask_RCNN --inplace --reportfile report.txt`
> 
> This will automatically update the existing code to TF2. You will also get a list of changes made in report.txt
> 
> **2. Replace the following line:**
> 
> `mrcnn_bbox = layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x) ` with this this if-else code block:
> 
> ```
> if s[1]==None:
>     mrcnn_bbox = layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
> else:
>     mrcnn_bbox = layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
> ```
> 
> **3. Change the following line:**
> 
> `indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1) ` with this line:
> 
> `indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis = 1)`
> 
> **4. Now, you need to replace:**
> 
> ` from keras import saving` with:
> 
> `from tensorflow.python.keras import saving ` then you will also want to replace the lines in both if and else block:
> 
> `saving.load_weights_from_hdf5_group(f, layers) ` and so on with the follwoing lines, inside if and else block respectively:
> 
> `saving.hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)`
> 
> `saving.hdf5_format.load_weights_from_hdf5_group(f, layers)`
>
> **5. Replace KE.Layer with Layer**
> 
> after adding `from tensorflow.keras.layers import Layer` to the preamble
>
> **6. Replace :**
> 
> `import keras.layers as KL` with `import tensorflow.keras.layers as KL`
>
> Thanks to: @deluongo, @Trotts, @nielsuit227, @ibrahimLearning @mayurmahurkar

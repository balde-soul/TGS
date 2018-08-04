# coding=utf-8
import tensorflow as tf
from colorama import Fore
import numpy as np
import logging
from collections import OrderedDict
import Putil.DenseNet.model_base as dmb
from tensorflow.contrib import layers
import Putil.np.util as npu
import Putil.tf.util as tfu


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)


def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")


def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        # subtract max is work for avoid overflow
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True, name='calc_max')
        exponential_map = tf.exp(tf.subtract(output_map, max_axis, 'sub_for_avoid_overflow'), 'exp')
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True, name='exp_sum')
        return tf.div(exponential_map, normalize, name='normalize')


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
                    summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        Fore.GREEN + "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: "
                     "{pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        with tf.name_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            conv1 = conv2d(in_node, w1, b1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):
        with tf.name_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
            bd = bias_variable([features // 2], name="bd")
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
            b1 = bias_variable([features // 2], name="b1")
            b2 = bias_variable([features // 2], name="b2")

            conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
            h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(h_conv, w2, b2, keep_prob)
            in_node = tf.nn.relu(conv2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

    # Output Map
    with tf.name_scope("output_map"):
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable([n_class], name="bias")
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        output_map = tf.nn.relu(conv)
        up_h_convs["out"] = output_map

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


def __reducor_for_DenseUNet(
        output_map,
        training,
        params
):
    param_dtype = tfu.tf_type(params.get('param_dtype')).Type
    regularize_weight = params.get('regularize_weight')
    grow = params.get('grows')
    kernel = params.get('kernels')
    layer_param = params.get('layer_param')
    layer_param['training'] = training
    output_map = dmb.DenseNetBlockLayers(
        output_map,
        param_dtype,
        grow,
        'reducor',
        regularize_weight,
        kernel,
        layer_param
    )
    return output_map
    pass


def __base_feature(
        output_map,
        params
):
    filter = params.get('feature_amount')
    kernel = params.get('kernel')
    stride = params.get('stride')
    param_dtype = tfu.tf_type(params.get('param_dtype')).Type
    regularize_weight = params.get('regularize_weight')

    output_map = tf.layers.conv2d(
        output_map,
        filter,
        kernel,
        stride,
        "same",
        activation=tf.nn.relu,
        kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
        kernel_regularizer=layers.l2_regularizer(regularize_weight),
        bias_initializer=tf.zeros_initializer(dtype=param_dtype),
        bias_regularizer=layers.l2_regularizer(regularize_weight),
        name='base'
    )
    return output_map
    pass


def __DenseUNet(
        output_map,
        training,
        DenseUNetConfig
):
    """

    :param output_map:
    :param training:
    :param DenseUNetConfig:
# {
#   "BaseModel": "DenseNet",
#   "BaseFeature":{
#     "feature_amount": 32,
#     "kernel": [3, 3],
#     "stride": [1, 1],
#     "param_dtype": 0.32,
#     "regularize_weight": 0.0001
#   },
#   "DenseNet":[
#     {
#       "param_dtype": 0.32,
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#       "pool_kernel": [2, 2],
#       "pool_stride": [2, 2],
#       "pool_type": "max",
#       "layer_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         }
#       },
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param": {
#           "type": "ReLU"
#         },
#         "compress_rate": null,
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#       "pool_kernel": [2, 2],
#       "pool_stride": [2, 2],
#       "pool_type": "max",
#       "layer_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         }
#       },
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param": {
#           "type": "ReLU"
#         },
#         "compress_rate": null,
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#       "pool_kernel": [2, 2],
#       "pool_stride": [2, 2],
#       "pool_type": "max",
#       "layer_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         }
#       },
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param": {
#           "type": "ReLU"
#         },
#         "compress_rate": null,
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#       "pool_kernel": [2, 2],
#       "pool_stride": [2, 2],
#       "pool_type": "max",
#       "layer_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         }
#       },
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param": {
#           "type": "ReLU"
#         },
#         "compress_rate": null,
#         "dropout_rate": 0.1
#       }
#     }
#   ],
#   "DeDenseNet":[
#     {
#       "param_dtype": 0.32,
#
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#
#       "t_kernel": [3, 3],
#       "t_stride": [2, 2],
#       "compress_rate": 0.3,
#
#       "layer_param":{
#         "batch_normal": true,
#         "activate":{
#           "type": "ReLU"
#         }
#       },
#
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         },
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#
#       "t_kernel": [3, 3],
#       "t_stride": [2, 2],
#       "compress_rate": 0.3,
#
#       "layer_param":{
#         "batch_normal": true,
#         "activate":{
#           "type": "ReLU"
#         }
#       },
#
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         },
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#
#       "t_kernel": [3, 3],
#       "t_stride": [2, 2],
#       "compress_rate": 0.3,
#
#       "layer_param":{
#         "batch_normal": true,
#         "activate":{
#           "type": "ReLU"
#         }
#       },
#
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         },
#         "dropout_rate": 0.1
#       }
#     },
#     {
#       "param_dtype": 0.32,
#
#       "grows": [3, 3, 3],
#       "regularize_weight": 0.0001,
#       "kernels": [[3, 3], [3, 3], [3, 3]],
#
#       "t_kernel": [3, 3],
#       "t_stride": [2, 2],
#       "compress_rate": 0.3,
#
#       "layer_param":{
#         "batch_normal": true,
#         "activate":{
#           "type": "ReLU"
#         }
#       },
#
#       "transition_param":{
#         "batch_normal": true,
#         "activate_param":{
#           "type": "ReLU"
#         },
#         "dropout_rate": 0.1
#       }
#     }
#   ],
#   "BlockReducor":{
#     "param_dtype": 0.32,
#     "regularize_weight": 0.0001,
#     "grows": [3, 2, 1],
#     "kernels": [[1, 1], [2, 2], [3, 3]],
#     "layer_param":{
#       "batch_normal": true,
#       "activate":{
#         "type": "ReLU"
#       }
#     }
#   }
# }

    :return:
    """
    BaseFeature = DenseUNetConfig.get('BaseFeature')
    DenseNetConfig = DenseUNetConfig.get('DenseNet')
    DeDenseNetConfig = DenseUNetConfig.get('DeDenseNet')
    BlockReducor = DenseUNetConfig.get('BlockReducor')

    output_map = __base_feature(output_map, BaseFeature)

    cl = dmb.DenseNetProvide()
    cld = dmb.DeDenseNetProvide()
    output_map = dmb.DenseNetFromParamDict(
        output_map,
        training,
        DenseNetConfig,
        dense_net_provide=cl,
        block_name_flag='encode-')
    block_layer_want = cl.BlockLayer[-1][-1]
    cl.BlockLayer.reverse()

    output_map = __reducor_for_DenseUNet(output_map, training, BlockReducor)

    de_block_name = 0
    for encode_block_layer in zip(cl.BlockLayer, DeDenseNetConfig):
        DeDenseNetBlockConfig = encode_block_layer[1]
        param_dtype = tfu.tf_type(DeDenseNetBlockConfig.get('param_dtype')).Type
        grows = DeDenseNetBlockConfig.get('grows')
        regularize_weight = DeDenseNetBlockConfig.get('regularize_weight')
        kernels = DeDenseNetBlockConfig.get('kernels')
        t_kernel = DeDenseNetBlockConfig.get('t_kernel')
        t_stride = DeDenseNetBlockConfig.get('t_stride')
        compress_rate = DeDenseNetBlockConfig.get('compress_rate')
        layer_param = DeDenseNetBlockConfig.get('layer_param')
        layer_param['training'] = training
        transition_param = DeDenseNetBlockConfig.get('transition_param')
        transition_param['training'] = training
        to_concat = encode_block_layer[0][-1]
        cld.push_block()
        output_map = dmb.DeDenseNetBlockTransition(
            output_map,
            param_dtype,
            'decode_{0}_{1}'.format(de_block_name, 'transition'),
            regularize_weight,
            t_kernel,
            t_stride,
            compress_rate,
            **transition_param
        )
        output_map = tf.concat(
            [to_concat, output_map],
            axis=-1,
            name='decode_{0}_{1}'.format(de_block_name, 'concat'))
        cld.push_transition(output_map)
        output_map = dmb.DeDenseNetBlockLayers(
            output_map,
            param_dtype,
            grows,
            'decode_{0}_{1}'.format(de_block_name, 'block_layer'),
            regularize_weight,
            kernels,
            layer_param,
        )
        cld.push_block_layer(output_map)
        de_block_name += 1
        pass
    return output_map
    pass


def DenseUNetPro(
        output_map,
        training,
        class_amount,
        param_dtype,
        regularizer_weight,
        DenseUNetConfig,
):
    output_map = __DenseUNet(
        output_map,
        training,
        DenseUNetConfig
    )
    output_map = __conv_pixel_wise_class_pro(
        output_map,
        class_amount,
        "fcn",
        param_dtype,
        regularizer_weight
    )
    # output_map = tf.reduce_max(output_map, axis=-1, keepdims=True, name='pixel_class')
    return output_map
    pass


# todo: calc the miou
def fcn_calc_miou(
        logit,
        gt
):
    pass


def __conv_pixel_wise_class_pro(output_map, class_amount, name, param_dtype, regularize_weight, **options):
    with tf.variable_scope('{0}_pixel_wise_class_pro'.format(name)):
        output_map = tf.layers.conv2d(
            output_map,
            filters=class_amount,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            bias_initializer=tf.variance_scaling_initializer(mode='fan_avg', dtype=param_dtype),
            kernel_regularizer=layers.l2_regularizer(regularize_weight),
            bias_regularizer=layers.l2_regularizer(regularize_weight),
            use_bias=True,
            name='conv'
        )
        with tf.variable_scope('ac'):
            alpha = tf.Variable(0.1, trainable=True)
            output_map = tf.nn.leaky_relu(output_map, alpha, name='PReLU')
            pass
        pass
    return output_map
    pass


def fcn_acc(logits, label):
    with tf.name_scope("fcn_acc"):
        pro = tf.arg_max(logits, -1)
        shape = tf.shape(pro)
        sub_shape = tf.slice(shape, [1], [-1])
        pixel_count = 0.5 * tf.cast((tf.square(tf.reduce_sum(sub_shape)) - tf.reduce_sum(sub_shape * sub_shape)), tf.float32)
        l = tf.arg_max(label, -1)
        no_zeros_count = tf.cast(tf.count_nonzero(pro - l, axis=-1), tf.float32)
        one_batch_sum = 1 - tf.reduce_sum(no_zeros_count, axis=-1) / pixel_count
        acc = tf.reduce_mean(one_batch_sum, axis=0)
        return acc
    pass


def fcn_loss(logits, label, cost_name, param_dtype, **options):
    """

    :param logits:
    :param label:
    :param cost_name:
    :param param_dtype:
    :param options:
    :return:
    """
    class_amount = logits.get_shape().as_list()[-1]
    with tf.name_scope("fcn_loss"):
        # flat_logits = tf.reshape(logits, [-1, class_amount])
        # flat_labels = tf.reshape(label, [-1, class_amount])
        flat_logits = logits
        flat_labels = label
        if cost_name == "cross_entropy":
            class_weights = options.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=npu.np_type(param_dtype).Type))

                # class weights is a 1-D array , here create the total weight map for weighting
                weight_map = tf.multiply(flat_labels, class_weights)
                # weight_map = tf.reduce_sum(weight_map, axis=-1)

                # calc entropy loss cross the dims[-1]
                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                      labels=flat_labels)

                # make weighting
                # weighted_loss = tf.multiply(loss_map, weight_map)

                weighted_loss = loss_map
                # loss = tf.reduce_mean(weighted_loss)
                loss = tf.reduce_mean(tf.reduce_mean(weighted_loss, axis=0))
            else:
                loss = tf.reduce_sum(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=flat_logits,
                        labels=flat_labels),
                        axis=0
                    ))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            # softmax the logits
            prediction = pixel_wise_softmax(logits)
            intersection = tf.reduce_sum(prediction * label)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(label)
            loss = -(2 * intersection / (union))

        else:
            raise ValueError("Unknown cost function: " % cost_name)
        return loss
    pass

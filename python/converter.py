"""
Reads Darknet config and weights and creates Keras model with TF backend.
"""

import argparse
import configparser
from contextlib import redirect_stdout
import os
import numpy as np
from keras import backend as K
from utils import parse_config_file , fold_batch_norm_layer , reorg_shape , reorg
from keras.layers import Conv2D, Input, Lambda, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate, Reshape, Activation, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU , Softmax
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def convert(config_path, weights_path, output_path, remove_batch_normalization = False, save_keras = True, save_tfjs = False, plot_model = False):

    model_name = os.path.splitext(config_path)[0]
    print('Parsing darknet config...')
    unique_config_file = parse_config_file(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)


  
    print('Loading weights...')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3, ), dtype='int32', buffer = weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer
    all_layers = []
    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4

    weight_count = 0 # a number to keep with the read
    out_index = []

    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))

        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            conv_bias = np.ndarray(shape=(filters, ),dtype='float32',buffer=weights_file.read(filters * 4))
            weight_count += filters

            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters),dtype='float32',buffer=weights_file.read(filters * 12))
                weight_count += 3 * filters
                 # scale gamma / # shift beta / # running mean /# running var
                bn_weight_list = [ bn_weights[0], conv_bias, bn_weights[1], bn_weights[2]]

            conv_weights = np.ndarray(shape=darknet_w_shape,dtype='float32',buffer=weights_file.read(weights_size * 4))
            weight_count += weights_size

            # Darknet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # we would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            if remove_batch_normalization:
                if batch_normalize:
                    conv_layer_weights = fold_batch_norm_layer(conv_weights,bn_weight_list)
                    
                else:
                    conv_layer_weights = [conv_weights, conv_bias]
                use_bias = True
            else:
                if batch_normalize:
                    conv_layer_weights = [conv_weights]
                    
                else:
                    conv_layer_weights = [conv_weights, conv_bias]
                use_bias = not batch_normalize 

            #########

            # Create Conv2D layer
            if stride>1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            
            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError('Unknown activation function `{}` in section {}'.format(activation, section))
 
            conv_layer = (Conv2D(filters, (size, size),strides=(stride, stride),kernel_regularizer=l2(weight_decay),use_bias=use_bias,weights=conv_layer_weights,activation=act_fn, padding=padding))(prev_layer)
            if  not remove_batch_normalization and batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
            
            prev_layer = conv_layer
            ###


            if activation == 'leaky':
                prev_layer = LeakyReLU(alpha=0.1)(prev_layer)
           
            all_layers.append(prev_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(MaxPooling2D(pool_size=(size, size),strides=(stride, stride),padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('reorg'):
            block_size = int(cfg_parser[section]['stride'])
            assert block_size == 2, 'Only reorg with stride 2 supported.'
            all_layers.append(Lambda(reorg,output_shape=reorg_shape,name=str(section))(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('softmax'):
            all_layers.append(Activation("softmax")(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('avgpool'):
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo') or section.startswith('region') :
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
            anchors = cfg_parser[section]['anchors']

        elif section.startswith('net'):
            pass

        else:
            raise ValueError('Unsupported section header type: {}'.format(section))

     # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(weight_count, weight_count + remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    # Create and save model.
    if len(out_index)==0: out_index.append(len(all_layers)-1)
    input_l = input_layer
    out_l = [all_layers[i] for i in out_index]

    model = Model(inputs=input_l, outputs=out_l)
    print(model.summary())
    
    if remove_batch_normalization:
        model_save_path = '{}/{}_nobn.h5'.format(output_path,model_name)
        model_summary_path = '{}/{}_nobn_summary.txt'.format(output_path,model_name)
        tfjs_path = '{}/{}-tfjs_nobn/'.format(output_path,model_name)
        plot_path = '{}/{}_nobn.png'.format(output_path,model_name)
        anchors_path = '{}/{}_nobn_anchors.txt'.format(output_path,model_name)

    else:
        model_save_path = '{}/{}.h5'.format(output_path,model_name)
        model_summary_path = '{}/{}_summary.txt'.format(output_path,model_name)
        tfjs_path = '{}/{}-tfjs/'.format(output_path,model_name)
        plot_path = '{}/{}.png'.format(output_path,model_name)
        anchors_path = '{}/{}_anchors.txt'.format(output_path,model_name)

  
            
    with open(anchors_path, 'w') as f:
        print(anchors, file=f)
        print('Saved anchors to {}'.format(anchors_path))

    
    with open(model_summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        print('Saved model summary to {}'.format(model_summary_path))


    if save_keras:
        model.save(model_save_path)
        print('Saved Keras model to {}'.format(model_save_path))
 
    if save_tfjs:
        import tensorflowjs as tfjs
        # tfjs.converters.save_keras_model(model, tfjs_path, quantization_dtype=np.uint8)
        tfjs.converters.save_keras_model(model, tfjs_path)
        print('Saved Tensorflowjs model to {}'.format(tfjs_path))

    if plot_model:
        from keras.utils.vis_utils import plot_model as plot
        plot(model, to_file=plot_path, show_shapes=True)
        print('Saved image plot to {}'.format(plot_path))


parser = argparse.ArgumentParser(description='Converts Darknet Models&Weights To Keras and Tensorflow.js Formats.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')

parser.add_argument('-b','--remove_batch_normalization',help='Fold the batch normalization parameters into the weights for the previous layers',action='store_true')

parser.add_argument('-tfjs','--save_tfjs',help='Fold the batch normalization parameters into the weights for the previous layers',action='store_true')
parser.add_argument('-keras','--save_keras',help='save as keras model',action='store_true')
parser.add_argument('-p','--plot_model',help='Plot generated Keras model and save as image.',action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    output_path = "converted" # output folder name 
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)
    convert(config_path, weights_path, output_path,args.remove_batch_normalization, args.save_keras, args.save_tfjs, args.plot_model)


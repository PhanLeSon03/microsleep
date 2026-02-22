# Generate a dataset of a numpy INPUT_SHAPE
# Save it to a TFRecord file such that at deserialization time we get a Tensor with all the float values.
# The objective is to avoid going via numpy serialization to bytes as that gets us to use py_function
# which is not supported TPU and doesn't get serialized to the TF computational graph.
#
# References
# [1] https://www.tensorflow.org/api_docs/python/tf/io/parse_single_sequence_example
# [2] https://colab.research.google.com/drive/1M10tbHih5eJ8LiApJSKKpNM79IconYJX#scrollTo=z1e5rhP6PTc3


import pandas as pd
import numpy as np
import tensorflow as tf
import os
from typing import Dict



class TFMultiOutParser():
    def __init__(self, shape_x, shape_y_dict: Dict):
        self.shape_x = shape_x
        x_proto = tf.io.FixedLenFeature(shape_x, dtype=tf.float32)
        self.features_descr = {'x': x_proto }

        for y in shape_y_dict:
            shape_y = shape_y_dict[y]
            y_proto = tf.io.FixedLenFeature(shape_y, dtype=tf.float32)
            self.features_descr[y] = y_proto

    @staticmethod
    def get_3out_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_3out

    def parse_3out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], (features['y1'], features['y2'], features['y3'])

    def get_1out3_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_1out3

    def parse_1out3(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], features['y3']

    @staticmethod
    def get_5out_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_5out

    def parse_5out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], (features['y1'], features['y2'], features['y3'], features['y4'], features['y5'])

    @staticmethod
    def get_2out_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_2out

    def parse_2out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], (features['y1'], features['y2'])

    @staticmethod
    def get_1out_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_1out

    def parse_1out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], features['y1'], 
    
    @staticmethod
    def get_1out_parser2(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParser(shape_x, shape_y_dict)
        return rp.parse_1out2

    def parse_1out2(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return features['x'], features['y2'], 

    @staticmethod
    def serialize_tfrecord(x: np.ndarray, y_dict: Dict):
        proto_x = tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten()))
        feature = {'x': proto_x}
        for y in y_dict:
            data = y_dict[y]
            proto_y = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
            feature[y] = proto_y

        sample_protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
        return sample_protobuf.SerializeToString()

    @staticmethod
    def write_tfrecord(out_file_path, feature_label_pair):
        if(os.path.exists(out_file_path)):
            os.remove(out_file_path)

        ops = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(out_file_path, options=ops) as protowriter:
            for x, y in feature_label_pair:
                sample = TFMultiOutParser.serialize_tfrecord(x, y)
                protowriter.write(sample)
            protowriter.flush()

'''=================================================================================================================='''
class TFMultiOutParserMIMO(TFMultiOutParser):
    def __init__(self, shape_x_dict: Dict, shape_y_dict: Dict):
        self.features_descr = {}
        for x in shape_x_dict:
            shape_x = shape_x_dict[x]
            x_proto = tf.io.FixedLenFeature(shape_x, dtype=tf.float32)
            self.features_descr[x] = x_proto
        for y in shape_y_dict:
            shape_y = shape_y_dict[y]
            y_proto = tf.io.FixedLenFeature(shape_y, dtype=tf.float32)
            self.features_descr[y] = y_proto

    @staticmethod
    def get_2in_2out_parser(shape_x_dict: Dict, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x_dict, shape_y_dict)
        return rp.parse_2in_2out

    @staticmethod
    def get_2in_1out_parser(shape_x_dict: Dict, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x_dict, shape_y_dict)
        return rp.parse_2in_1out

    @staticmethod
    def get_2out_parser(shape_x, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x, shape_y_dict)
        return rp.parse_2out
    
    @staticmethod
    def serialize_tfrecord_1(x: np.ndarray, y_dict: Dict):
        proto_x = tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten()))
        feature = {'x': proto_x}
        for y in y_dict:
            data = y_dict[y]
            proto_y = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
            feature[y] = proto_y

        sample_protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
        return sample_protobuf.SerializeToString()
    

    @staticmethod
    def serialize_tfrecord_2(x: np.ndarray, y_dict: Dict):
        proto_x1 = tf.train.Feature(float_list=tf.train.FloatList(value=x[0].flatten()))
        proto_x2 = tf.train.Feature(float_list=tf.train.FloatList(value=x[1].flatten()))
        feature = {'x1': proto_x1, 'x2': proto_x2}
        for y in y_dict:
            data = y_dict[y]
            proto_y = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
            feature[y] = proto_y

        sample_protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
        return sample_protobuf.SerializeToString()

    @staticmethod
    def serialize_tfrecord_segment(x: np.ndarray, y_dict: Dict):
        proto_x1 = tf.train.Feature(float_list=tf.train.FloatList(value=x[0].flatten()))
        proto_x2 = tf.train.Feature(float_list=tf.train.FloatList(value=x[1].flatten()))
        feature = {'x1': proto_x1, 'x2': proto_x2}
        for y in y_dict:
            data = y_dict[y]
            proto_y = tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
            feature[y] = proto_y

        sample_protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
        return sample_protobuf.SerializeToString()

    def parse_2in_2out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return (features['x1'], features['x2']), (features['y1'], features['y2'])

    def parse_2in_1out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return (features['x1'], features['x2']), features['y']

    @staticmethod
    def get_2in_3out_parser(shape_x_dict: Dict, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x_dict, shape_y_dict)
        return rp.parse_2in_3out

    def parse_2in_3out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return (features['x1'], features['x2']), (features['y1'], features['y2'], features['y3'])

    
    def parse_2in_2out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return (features['x1'], features['x2']), (features['y1'], features['y2'])
    
    
    @staticmethod
    def get_1in_3out_parser(shape_x_dict: Dict, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x_dict, shape_y_dict)
        return rp.parse_1in_3out

    def parse_1in_3out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        return (features['x1']), (features['y1'], features['y2'], features['y3'])
    
    @staticmethod
    def write_tfrecord_2inputs(out_file_path, feature_label_pair):
        if (os.path.exists(out_file_path)):
            os.remove(out_file_path)

        ops = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(out_file_path, options=ops) as protowriter:
            for x, y in feature_label_pair:
                sample = TFMultiOutParserMIMO.serialize_tfrecord_2(x, y)
                protowriter.write(sample)
            protowriter.flush()
            
    @staticmethod
    def write_tfrecord_1input(out_file_path, feature_label_pair):
        if (os.path.exists(out_file_path)):
            os.remove(out_file_path)

        ops = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(out_file_path, options=ops) as protowriter:
            for x, y in feature_label_pair:
                sample = TFMultiOutParserMIMO.serialize_tfrecord_1(x, y)
                protowriter.write(sample)
            protowriter.flush()
            
            

    @staticmethod
    def get_2in_out_parser(shape_x_dict: Dict, shape_y_dict: Dict):
        rp = TFMultiOutParserMIMO(shape_x_dict, shape_y_dict)
        return rp.parse_2in_out
    
    def parse_2in_out(self, proto):
        features = tf.io.parse_single_example(proto, features=self.features_descr)
        print(features['y1'].shape)
        return (features['x1'], features['x2']), (features['y1'])




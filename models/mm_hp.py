import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from keras import Model
from keras import optimizers
from kerastuner import HyperModel
from transformers import TFBertModel
from keras.applications import ResNet101, Xception
from efficientnet.keras import EfficientNetB0
from config import *

grid_models = {
    "Resnet": ResNet101,
    "Xception": Xception,
    "EfficientNet": EfficientNetB0
}


def build_txtEncoder():
    bert = TFBertModel.from_pretrained(TEXTMODEL)
    return bert


def build_imgEncoder(grid_name):
    # conv_base model
    base_model = grid_models[grid_name](weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x_out = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    conv_base = Model(inputs=base_model.input, outputs=x_out, name="ConvBase")
    return conv_base


class HyperModelConcat(HyperModel):
    def __init__(self, grid_name=''):
        self.num_labels = NUM_LABELS
        self.grid_name = grid_name

    def build(self, hp):
        MODEL = "MM-{}-CONCAT".format(self.grid_name)
        bert = build_txtEncoder()
        conv_base = build_imgEncoder(self.grid_name)
        # text inputs# INPUT
        in_id = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="input_ids")
        in_mask = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="attention_mask")
        in_segment = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="token_type_ids")
        # img inputs
        in_img = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3,), dtype="float32", name="input_imgs")
        inputs = {"input_ids": in_id,
                  "attention_mask": in_mask,
                  "token_type_ids": in_segment,
                  "img_input": in_img
                  }
        inputs_bert = inputs.copy()
        img_input = inputs_bert.pop("img_input")
        # out of img is 2048 and text is 768
        bert_out = bert(inputs_bert, return_dict=True)
        txt = bert_out.pooler_output
        img = conv_base(img_input)
        # project the img to a 768 space (same dimension as bert)
        img_dense = Dense(HFixed, name="ImgDense")(img)
        # concatenate layers
        H_fused = tf.keras.layers.Concatenate()([txt, img_dense])
        # H_dense = Dense(hp.Choice('NumHD', values=[100, 200, 400]), activation = "relu", name = "HDense")(H_fused)
        x = Dropout(rate=hp.Float(
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
            name="dropout_rate"
        ), name="top_dropout")(H_fused)
        output = Dense(NUM_LABELS, activation="softmax", name="pred")(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=output, name=MODEL)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                                          values=[1e-3, 1e-4, 1e-5])),
                      metrics=[tf.keras.metrics.AUC(), "accuracy"])
        return model

class HyperModelAttM(HyperModel):
    def __init__(self, grid_name=''):
        self.num_labels = NUM_LABELS
        self.grid_name = grid_name

    def build(self, hp):
        MODEL = "MM-{}-ATTM".format(self.grid_name)
        bert = build_txtEncoder()
        conv_base = build_imgEncoder(self.grid_name)
        # text inputs# INPUT
        in_id = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="input_ids")
        in_mask = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="attention_mask")
        in_segment = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="token_type_ids")
        # img inputs
        in_img = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3,), dtype="float32", name="input_imgs")
        inputs = {"input_ids": in_id,
                  "attention_mask": in_mask,
                  "token_type_ids": in_segment,
                  "img_input": in_img
                  }
        inputs_bert = inputs.copy()
        img_input = inputs_bert.pop("img_input")
        # out of img is 2048 and text is 768
        bert_out = bert(inputs_bert, return_dict=True)
        txt = bert_out.pooler_output
        img = conv_base(img_input)
        # project the img to a 768 space
        img_proj = Dense(HFixed, name="ImgProj")(img)
        # project the img and text to a 200 space
        fixed = 200
        img_dense = Dense(fixed, name="ImgDense", activation=tf.nn.tanh)(img_proj)
        txt_dense = Dense(fixed, name="TxtDense", activation=tf.nn.tanh)(txt)
        # scale matmul_qk
        dk = tf.cast(tf.shape(txt)[-1], tf.float32)
        matmul_qk = tf.matmul(txt, img_proj, transpose_b=True)
        scaled_att = matmul_qk / tf.math.sqrt(dk)
        alpha = tf.nn.softmax(scaled_att)
        H_fused = tf.matmul(alpha, txt_dense) + tf.matmul(1 - alpha, img_dense)

        x = Dropout(rate=hp.Float(
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
            name="dropout_rate"
        ), name="top_dropout")(H_fused)
        output = Dense(NUM_LABELS, activation="softmax", name="pred")(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=output, name=MODEL)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                                          values=[1e-3, 1e-4, 1e-5])),
                      metrics=[tf.keras.metrics.AUC(), "accuracy"])
        return model

class HyperModelGLU(HyperModel):
    def __init__(self, grid_name=''):
        self.num_labels = NUM_LABELS
        self.grid_name = grid_name

    def build(self, hp):
        MODEL = "MM-{}-ATT".format(self.grid_name)
        fixed = 200
        bert = build_txtEncoder()
        conv_base = build_imgEncoder(self.grid_name)
        # text inputs
        in_id = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="input_ids")
        in_mask = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="attention_mask")
        in_segment = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="token_type_ids")
        # img inputs
        in_img = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3,),dtype="float32", name="input_imgs")
        inputs = {  "input_ids":in_id,
                    "attention_mask": in_mask,
                    "token_type_ids": in_segment,
                    "img_input":in_img
                    }
        inputs_bert = inputs.copy()
        img_input = inputs_bert.pop("img_input")
        # out of img is 2048 and text is 768
        bert_out = bert(inputs_bert, return_dict=True)
        txt = bert_out.pooler_output
        img = conv_base(img_input)
        #project the img and text to a 200 space
        img_dense = Dense(fixed, name="ImgDense", activation=tf.nn.tanh)(img)
        txt_dense = Dense(fixed,name = "TxtDense", activation=tf.nn.tanh)(txt)

        # concat txt and img
        txt_img = tf.keras.layers.Concatenate(axis=-1)([txt,img])
        gate_z = Dense(fixed, activation=tf.nn.sigmoid, name = "Gate")(txt_img)
        H_fused = gate_z * txt_dense + (1 - gate_z) * img_dense

        x = Dropout(rate=hp.Float(
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
            name="dropout_rate"
        ), name="top_dropout")(H_fused)
        output = Dense(NUM_LABELS, activation="softmax", name="pred")(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=output, name=MODEL)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                                          values=[1e-3, 1e-4, 1e-5])),
                      metrics=[tf.keras.metrics.AUC(), "accuracy"])
        return model

class HyperModelXAtt(HyperModel):
    def __init__(self, grid_name=''):
        self.num_labels = NUM_LABELS
        self.grid_name = grid_name

    def build(self, hp):
        MODEL = "MM-{}-ATT".format(self.grid_name)
        bert = build_txtEncoder()
        conv_base = build_imgEncoder(self.grid_name)
        # text inputs
        in_id = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="input_ids")
        in_mask = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="attention_mask")
        in_segment = tf.keras.Input(shape=(MAX_SEQ,), dtype="int32", name="token_type_ids")
        # img inputs
        in_img = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3,), dtype="float32", name="input_imgs")
        inputs = {"input_ids": in_id,
                  "attention_mask": in_mask,
                  "token_type_ids": in_segment,
                  "img_input": in_img
                  }
        inputs_bert = inputs.copy()
        img_input = inputs_bert.pop("img_input")
        # out of img is 2048 and text is 768
        bert_out = bert(inputs_bert, return_dict=True)
        txt = bert_out.pooler_output
        img = conv_base(img_input)
        img_dense = Dense(HFixed, name="ImgDense")(img)
        # attention
        H_txt = tf.keras.layers.Attention()([txt, img_dense]) #, return_attention_scores=True)
        H_img = tf.keras.layers.Attention()([img_dense,txt])
        H_fused = tf.keras.layers.Add()([H_txt,H_img])

        x = Dropout(rate=hp.Float(
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
            name="dropout_rate"
        ), name="top_dropout")(H_fused)
        output = Dense(NUM_LABELS, activation="softmax", name="pred")(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=output, name=MODEL)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                                          values=[1e-3, 1e-4, 1e-5])),
                      metrics=[tf.keras.metrics.AUC(), "accuracy"])
        return model
class HyperModelGLUATT(HyperModel):
    def __init__(self, grid_name=''):
        self.num_labels = NUM_LABELS
        self.grid_name = grid_name

    def build(self, hp):
        MODEL = "MM-{}-GLUATT".format(self.grid_name)
        fixed = 200
        bert = build_txtEncoder()
        conv_base = build_imgEncoder(self.grid_name)
        # text inputs
        in_id = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="input_ids")
        in_mask = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="attention_mask")
        in_segment = tf.keras.Input(shape=(MAX_SEQ,),dtype="int32", name="token_type_ids")
        # img inputs
        in_img = tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3,),dtype="float32", name="input_imgs")
        inputs = {  "input_ids":in_id,
                    "attention_mask": in_mask,
                    "token_type_ids": in_segment,
                    "img_input":in_img
                    }
        inputs_bert = inputs.copy()
        img_input = inputs_bert.pop("img_input")
        # out of img is 2048 and text is 768
        bert_out = bert(inputs_bert, return_dict=True)
        txt = bert_out.pooler_output
        img = conv_base(img_input)
        #project the img and text to a 200 space
        img_dense = Dense(fixed, name="ImgDense", activation=tf.nn.tanh)(img)
        txt_dense = Dense(fixed,name = "TxtDense", activation=tf.nn.tanh)(txt)

        # concat txt and img
        txt_img = tf.keras.layers.Concatenate(axis=-1)([txt,img])
        gate_z = Dense(fixed, activation=tf.nn.sigmoid, name = "Gate")(txt_img)
        G_txt = gate_z * txt_dense
        G_img = (1 - gate_z) * img_dense

        # attention
        H_txt = tf.keras.layers.Attention()([G_txt, G_img])  # , return_attention_scores=True)
        H_img = tf.keras.layers.Attention()([G_img, G_txt])
        H_fused = tf.keras.layers.Add()([H_txt, H_img])

        x = Dropout(rate=hp.Float(
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
            name="dropout_rate"
        ), name="top_dropout")(H_fused)
        output = Dense(NUM_LABELS, activation="softmax", name="pred")(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=output, name=MODEL)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(hp.Choice('learning_rate',
                                                          values=[1e-3, 1e-4, 1e-5])),
                      metrics=[tf.keras.metrics.AUC(), "accuracy"])
        return model



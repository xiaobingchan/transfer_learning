import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

CKPT_DIR = './ckpt/'
DATASET_DIR = './dataset/train/'
CSV_PATH = './dataset/train.csv'
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 5e-6
EPOCHS = 100
BATCH_SIZE = 4
NUM_OF_CLASSES = 6
MODEL_IMAGE_SIZE = (224, 224, 3)
PROCESS_IMAGE_SIZE = (224, 224)
CALLBACKS = [
    ModelCheckpoint(filepath='./ckpt/model.{epoch:04d}-{val_accuracy:.04f}.h5',  # 文件夹要提前自建
                    monitor='val_accuracy',
                    verbose=1,
                    period=2),
    TensorBoard(log_dir='./logs/{}'.format('train'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'),
    EarlyStopping(monitor='val_accuracy',
                  min_delta=1e-4,
                  patience=5,
                  verbose=1),
]


def set_gpus():
    """设置物理GPU选项"""
    gpus = tf.config.list_physical_devices(device_type='GPU')
    if len(gpus) != 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)


def efficientnet(input_shape, num_of_classes):
    """构建模型"""

    input_layer = Input(shape=input_shape, name='Input-Layer')
    efficient_layer = EfficientNetB4(include_top=False,
                                     weights='imagenet',
                                     input_tensor=input_layer)(input_layer)
    flatten_layer = Flatten(name='Flatten-Layer')(efficient_layer)
    dense_layer = Dense(units=128, activation='relu', name='Dense-Layer')(flatten_layer)
    dropout_layer = Dropout(rate=0.5, name='Dropout-Layer')(dense_layer)
    output_layer = Dense(units=num_of_classes, activation='softmax', name='Output-Layer')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def get_dataframe(csv_path):
    """得到dataframe"""
    dataframe = pd.read_csv(csv_path, index_col=None)
    dataframe['label'] = dataframe['label'].astype(str)
    dataframe = dataframe.sample(frac=1)

    return dataframe


def load_preprocess(image_size, batch_size, dataset_dir, csv_path, validation_split=0.2):
    """对图像进行预处理"""
    datagen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 channel_shift_range=10,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    df = get_dataframe(csv_path)
    if validation_split > 0:
        train_indices = int((1 - validation_split) * df.shape[0])
        train_df = df.iloc[0:train_indices]
        validation_df = df.iloc[train_indices:]

        train_batches = datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=dataset_dir,
                                                    x_col='filename',
                                                    y_col='label',
                                                    target_size=image_size,
                                                    interpolation='bicubic',
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    validate_filenames=True)
        validation_batches = datagen.flow_from_dataframe(dataframe=validation_df,
                                                         directory=dataset_dir,
                                                         x_col='filename',
                                                         y_col='label',
                                                         target_size=image_size,
                                                         interpolation='bicubic',
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         batch_size=batch_size,
                                                         validate_filenames=True)
        return train_batches, validation_batches
    else:
        batches = datagen.flow_from_dataframe(dataframe=df,
                                              directory=dataset_dir,
                                              x_col='filename',
                                              y_col='label',
                                              target_size=image_size,
                                              interpolation='bicubic',
                                              class_mode='categorical',
                                              shuffle=True,
                                              batch_size=batch_size,
                                              validate_filenames=True)

        return batches


if __name__ == '__main__':
    # 设置GPU显存
    set_gpus()

    # 实例化模型
    if os.path.exists(CKPT_DIR) is False:
        os.mkdir(CKPT_DIR)
        model = efficientnet(MODEL_IMAGE_SIZE, 6)
    else:
        print('continue..........')
        file_list = os.listdir(CKPT_DIR)
        model = load_model(CKPT_DIR + file_list[np.argmax(file_list)])
    model.summary()
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 加载数据
    batches = load_preprocess(PROCESS_IMAGE_SIZE, BATCH_SIZE, DATASET_DIR, CSV_PATH, 0.1)
    print(batches[0].class_indices)
    print(batches[1].class_indices)

    try:
        # 训练模型并保存
        model.fit(x=batches[0],
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=1,
                  callbacks=CALLBACKS,
                  validation_data=batches[1])
        model.save('./model.h5')
    except KeyboardInterrupt:
        # 提前结束也会保存一个模型
        model.save('./model.h5')
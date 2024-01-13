# 导入所需的库
from tensorflow import lite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random, os
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv(r'../input/diabetic-retinopathy-224x224-gaussian-filtered/train.csv')

# 定义用于将多类别标签映射为二进制标签的字典
diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

# 定义用于将多类别标签映射为具体标签的字典
diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

# 添加二进制标签列和具体标签列到数据框
df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)

# 打印数据框的前几行
df.head()

plt.figure(figsize=(8, 6))
df['type'].value_counts().plot(kind='barh')
plt.title('Label Distribution')
plt.xlabel('Count')
plt.ylabel('Type')

plt.savefig('label_distribution1.png')

plt.show()
train_intermediate, val = train_test_split(df, test_size = 0.15, stratify = df['type'])
train, test = train_test_split(train_intermediate, test_size = 0.15 / (1 - 0.15), stratify = train_intermediate['type'])

print("For Training Dataset :")
print(train['type'].value_counts(), '\n')
print("For Testing Dataset :")
print(test['type'].value_counts(), '\n')
print("For Validation Dataset :")
print(val['type'].value_counts(), '\n')

import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 训练集柱状图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
train['type'].value_counts().plot(kind='bar')
plt.title('Training Dataset')
plt.xlabel('Type')
plt.ylabel('Count')

# Test set bar chart
plt.subplot(1, 3, 2)
test['type'].value_counts().plot(kind='bar')
plt.title('Test Dataset')
plt.xlabel('Type')
plt.ylabel('Count')

# Validation set bar chart
plt.subplot(1, 3, 3)
val['type'].value_counts().plot(kind='bar')
plt.title('Validation Dataset')
plt.xlabel('Type')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('type_count2.png')
plt.show()
base_dir = ''

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)
# 源文件目录
src_dir = r'../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images'

# 复制训练集图像到目标文件夹
for index, row in train.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"

    # 构建源文件路径
    srcfile = os.path.join(src_dir, diagnosis, id_code)

    # 构建目标文件夹路径
    dstfile = os.path.join(train_dir, binary_diagnosis)

    # 输出路径信息
    print(f"Copying {srcfile} to {dstfile}")

    # 创建目标文件夹
    os.makedirs(dstfile, exist_ok=True)

    # 复制图像文件
    shutil.copy(srcfile, dstfile)

# 复制验证集图像到目标文件夹
for index, row in val.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"

    # 构建源文件路径
    srcfile = os.path.join(src_dir, diagnosis, id_code)

    # 构建目标文件夹路径
    dstfile = os.path.join(val_dir, binary_diagnosis)

    # 输出路径信息
    print(f"Copying {srcfile} to {dstfile}")

    # 创建目标文件夹
    os.makedirs(dstfile, exist_ok=True)

    # 复制图像文件
    shutil.copy(srcfile, dstfile)

# 复制测试集图像到目标文件夹
for index, row in test.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"

    # 构建源文件路径
    srcfile = os.path.join(src_dir, diagnosis, id_code)

    # 构建目标文件夹路径
    dstfile = os.path.join(test_dir, binary_diagnosis)

    # 输出路径信息
    print(f"Copying {srcfile} to {dstfile}")

    # 创建目标文件夹
    os.makedirs(dstfile, exist_ok=True)

    # 复制图像文件
    shutil.copy(srcfile, dstfile)
train_path = 'train'
val_path = 'val'
test_path = 'test'

train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train_path, target_size=(224,224), shuffle = True)
val_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(val_path, target_size=(224,224), shuffle = True)
test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path, target_size=(224,224), shuffle = False)

import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import plot_model

def build_cnn_model(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.15):
    """
    构建一个简单的CNN模型。

    参数:
    - input_shape: 输入图像的形状 (默认为224x224x3)
    - num_classes: 输出的类别数量 (默认为2)
    - dropout_rate: Dropout层的丢失率 (默认为0.15)

    返回:
    一个已经编译的CNN模型
    """
    model = tf.keras.Sequential([
        layers.Conv2D(8, (3, 3), padding="valid", input_shape=input_shape, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(16, (3, 3), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(32, (4, 4), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (4, 4), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
        # 保存模型结构图
    plot_model(
        model,
        to_file='model_cnn.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    model.save('64x3-CNN.model')
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['acc'])

    return model

# 使用函数创建模型
model = build_cnn_model()

# 显示模型的摘要
model.summary()

history = model.fit(train_batches,
                    epochs=100,
                    validation_data=val_batches)

import matplotlib.pyplot as plt

# 训练过程中的损失和准确率
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']

# 生成 loss 和 val_loss 在一起的图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 生成准确率图
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('Training loss+Accuracy.png')
# 显示图形
plt.show()


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 预测概率
y_pred_prob = model.predict(val_batches)

# 获取真正率、假正率和阈值
fpr, tpr, thresholds = roc_curve(val_batches.classes, y_pred_prob[:, 1])

# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc Training.png")
plt.show()

# 计算混淆矩阵
y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)
cm = confusion_matrix(val_batches.classes, y_pred)

# 计算精确率、召回率、F1分数
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("Confusion Matrix Training.png")
plt.show()

# 绘制精确率-召回率曲线
precision, recall, thresholds = precision_recall_curve(val_batches.classes, y_pred_prob[:, 1])
average_precision = average_precision_score(val_batches.classes, y_pred_prob[:, 1])

plt.figure(figsize=(8, 8))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
plt.savefig('Training 评价指标.png')
plt.show()

# 使用函数创建模型
model = build_cnn_model()

# 显示模型的摘要
model.summary()

history = model.fit(test_batches,
                    epochs=100,
                    validation_data=val_batches)

import matplotlib.pyplot as plt

# 训练过程中的损失和准确率
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']

# 生成 loss 和 val_loss 在一起的图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Testing and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 生成准确率图
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('Testing loss+Accuracy.png')
# 显示图形
plt.show()


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 预测概率
y_pred_prob = model.predict(val_batches)

# 获取真正率、假正率和阈值
fpr, tpr, thresholds = roc_curve(val_batches.classes, y_pred_prob[:, 1])

# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc Testing.png")
plt.show()

# 计算混淆矩阵
y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)
cm = confusion_matrix(val_batches.classes, y_pred)

# 计算精确率、召回率、F1分数
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# 绘制混淆矩阵
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("Confusion Matrix Testing.png")
plt.show()

# 绘制精确率-召回率曲线
precision, recall, thresholds = precision_recall_curve(val_batches.classes, y_pred_prob[:, 1])
average_precision = average_precision_score(val_batches.classes, y_pred_prob[:, 1])

plt.figure(figsize=(8, 8))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(f'Precision-Recall Curve: AP={average_precision:.2f}')
plt.savefig('Testing 评价指标.png')
plt.show()
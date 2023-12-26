import os
import tarfile
from google.colab import drive
import split_folders
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.xception import Xception
import matplotlib.pyplot as plt

# Montar o Google Drive
drive.mount('/content/gdrive/')

# Criar um diretório chamado 'dataset'
os.mkdir('dataset')

# Divisão dos dados em treino, validação e teste
imgpath = './images'
output_ = './dataset'

# Dividir os dados com uma proporção (80% treino, 10% validação, 10% teste)
split_folders.ratio(imgpath, output=output_, seed=1337, ratio=(.80, .1, .1))

# Diretórios dos dados
train_dir = 'gdrive/My Drive/dataset/train'
test_dir = 'gdrive/My Drive/dataset/test'
vali_dir = 'gdrive/My Drive/dataset/val'

# Pré-processamento dos dados de treino, validação e teste
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)
vali_datagen = ImageDataGenerator(rescale=1./255)

# Criar geradores de dados de treino, validação e teste
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode="sparse"
)

validation_generator = vali_datagen.flow_from_directory(
    vali_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode="sparse"
)

test_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode="sparse"
)

# Construir um modelo usando a arquitetura Xception da Keras
xmodel = Xception(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

# Compilar o modelo
xmodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(),
    metrics=['accuracy']
)

# Treinar o modelo
xhistory = xmodel.fit_generator(
    train_generator,
    steps_per_epoch=None,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=None
)

# Salvar o modelo e seus pesos
xmodel.save('/content/gdrive/My Drive/xmodel.h5')
xmodel.save_weights('/content/gdrive/My Drive/xmodel_weight.h5')

# Avaliar o modelo no conjunto de teste
eva = xmodel.evaluate(test_generator)
print(eva)

# Plotar a precisão do modelo no conjunto de treino e validação
plt.plot(xhistory.history['accuracy'])
plt.plot(xhistory.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

# Carregar o modelo com pesos predefinidos
from keras.models import load_model
xmodel = load_model('/content/gdrive/My Drive/xmodel.h5')

# Avaliar o modelo no conjunto de teste
eva = xmodel.evaluate(test_generator)
print(eva)

# Pré-processamento de dados e criação de um novo modelo
xmodel = keras.applications.xception.Xception(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

sgdn = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

xmodel.trainable = False

xmodel.layers.pop()

xmodel_1 = keras.models.Sequential([
    xmodel,
    keras.layers.Dense(2048, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(120, activation="softmax")
])

# Compilar o modelo
xmodel_1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=sgdn,
    metrics=['accuracy']
)

# Treinar o modelo
xhistory_1 = xmodel_1.fit_generator(
    train_generator,
    steps_per_epoch=None,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=None
)

# Salvar o modelo e seus pesos
xmodel_1.save('/content/gdrive/My Drive/xmodel_1.h5')
xmodel_1.save_weights('/content/gdrive/My Drive/xmodel_1_weight.h5')

# Plotar a precisão do modelo no conjunto de treino e validação
plt.plot(xhistory_1.history['accuracy'])
plt.plot(xhistory_1.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

# Carregar o modelo com pesos predefinidos
xmodel_1 = load_model('/content/gdrive/My Drive/xmodel_1.h5')

# Avaliar o modelo no conjunto de teste
eva = xmodel_1.evaluate(test_generator)
print(eva)

# Pré-processamento de dados e criação de um novo modelo
xmodel = keras.applications.xception.Xception(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

xmodel.trainable = False

xmodel.layers.pop()

xmodel_2 = keras.models.Sequential([
    xmodel,
    keras.layers.Dense(2048, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(120, activation="softmax")
])

# Compilar o modelo
xmodel_2.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=sgdn,
    metrics=['accuracy']
)

# Treinar o modelo
xhistory_3 = xmodel_2.fit_generator(
    train_generator,
    steps_per_epoch=None,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=None
)

# Salvar o modelo e seus pesos
xmodel_2.save('/content/gdrive/My Drive/xmodel_2.h5')
xmodel_2.save_weights('/content/gdrive/My Drive/xmodel_2_weight.h5')

# Plotar a precisão do modelo no conjunto de treino e validação
plt.plot(xhistory_3.history['accuracy'])
plt.plot(xhistory_3.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

# Carregar o modelo com pesos predefinidos
xmodel_2 = load_model('/content/gdrive/My Drive/xmodel_2.h5')

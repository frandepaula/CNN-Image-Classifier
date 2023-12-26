import os
from google.colab import drive
import split_folders
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras import models
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
    test_dir,
    target_size=(299, 299),
    batch_size=20,
    class_mode="sparse"
)

# Construir um modelo usando a arquitetura Inception V3 da Keras
imodel = InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=(299, 299, 3),
    pooling=None,
    classes=1000
)

# Compilar o modelo
imodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.SGD(),
    metrics=['accuracy']
)

# Treinar o modelo
ihistory = imodel.fit_generator(
    train_generator,
    steps_per_epoch=None,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=None
)

# Salvar o modelo e seus pesos
imodel.save('/content/gdrive/My Drive/imodel.h5')
imodel.save_weights('/content/gdrive/My Drive/imodel_weight.h5')

# Avaliar o modelo no conjunto de teste
eva = imodel.evaluate(test_generator)
print(eva)

# Plotar a precisão do modelo no conjunto de treino e validação
plt.plot(ihistory.history['accuracy'])
plt.plot(ihistory.history['val_accuracy'])
plt.title('Precisão do Modelo')
plt.ylabel('Precisão')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper left')
plt.show()

# Carregar o modelo com pesos predefinidos
from keras.models import load_model
imodel = load_model('/content/gdrive/My Drive/imodel.h5')

# Avaliar o modelo no conjunto de teste
eva = imodel.evaluate(test_generator)
print(eva)


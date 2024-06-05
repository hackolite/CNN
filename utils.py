import os
import urllib.request
import gzip
import shutil
import numpy as np

# URLs alternatives des fichiers MNIST
urls = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
}


# Répertoire de destination pour les fichiers téléchargés et extraits
data_dir = './data'
#os.makedirs(data_dir, exist_ok=True)



def load_images(file_path):
    with open(file_path, 'rb') as f:
        _ = f.read(16)  # Sauter l'en-tête
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        _ = f.read(8)  # Sauter l'en-tête
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def download_and_extract(url, output_path):
    gz_path = output_path + '.gz'
    print(f'Téléchargement de {url}...')
    urllib.request.urlretrieve(url, gz_path)
    print(f'Extraction de {gz_path}...')
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f'Fichier extrait : {output_path}')


def get_data():
    data_dir = './data'
    train_images_path = os.path.join(data_dir, 'train_images.idx')
    train_labels_path = os.path.join(data_dir, 'train_labels.idx')
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    return (train_images, train_labels)


# Télécharger et extraire chaque fichier
#for key, url in urls.items():
#    print(url)
#    output_path = os.path.join(data_dir, key + '.idx')
#    download_and_extract(url, output_path)
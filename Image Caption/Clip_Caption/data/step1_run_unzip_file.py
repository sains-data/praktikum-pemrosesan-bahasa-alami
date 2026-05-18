import zipfile
import tarfile


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('corpus')


def untar(folder):
    t = tarfile.open(folder)
    t.extractall(path = 'corpus')
    t.close()


if __name__ == '__main__':
    path = './corpus/flickr30k-images.tar.gz'
    untar(path)
    print('Ekstrak dataset Flickr selesai.')
    
    
    train_data_path = './corpus/ai_challenger_caption_train_20170902'
    extract(train_data_path)
    print('Ekstrak set latih selesai.')
    
    
    valid_data_path = './corpus/ai_challenger_caption_validation_20170910'
    extract(valid_data_path)
    print('Ekstrak set validasi selesai.')
    
    
    

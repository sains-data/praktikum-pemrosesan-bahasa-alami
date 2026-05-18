"""
OCR pada frame video dengan PaddleHub (model Chinese DB+CRNN).
Mata kuliah NLP Berbasis Transformer, Itera.

Instalasi:
  pip install paddlepaddle==2.2.2 paddlehub==2.0.0 opencv-python
  pip install shapely==1.8.1.post1 pyclipper==1.3.0.post2
"""
import os
import time

import cv2
import paddlehub as hub


if __name__ == '__main__':
    frame_dir = './frame'
    if not os.path.isdir(frame_dir) or not os.listdir(frame_dir):
        raise FileNotFoundError(
            f'Folder "{frame_dir}" kosong. Jalankan step1_extract_frame.py terlebih dahulu.'
        )

    # Model server — akurasi lebih tinggi (butuh resource lebih besar)
    ocr = hub.Module(name='chinese_ocr_db_crnn_server')

    files = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.png')))
    image_paths = [os.path.join(frame_dir, f) for f in files]
    print(f'Jumlah gambar: {len(image_paths)}')

    images = [cv2.imread(p) for p in image_paths]
    images = [img for img in images if img is not None]

    start = time.time()
    results = ocr.recognize_text(
        images=images,
        use_gpu=False,
        output_dir='ocr_result',
        visualization=False,
        box_thresh=0.5,
        text_thresh=0.5,
    )
    elapsed = time.time() - start

    final_text = []
    for res in results:
        try:
            final_text.append(res['data'][-1]['text'])
        except (KeyError, IndexError, TypeError):
            continue

    print('Teks per frame:', final_text)
    unique_text = list(dict.fromkeys(final_text))
    merged = '，'.join(unique_text)
    print('Teks unik (gabungan):', merged)
    print(f'Total waktu: {elapsed:.2f} detik')

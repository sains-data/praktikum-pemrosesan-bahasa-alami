"""
Augmentasi data dengan back-translation memakai Baidu Translate API.
Format input: q_gov.txt — satu baris per pasangan teks||jawaban
"""
import random
import hashlib
import requests
from tqdm import tqdm


def baidu_translate(content, appid, secretKey, t_from='en', t_to='zh'):
    """Memanggil API terjemahan Baidu (maks. 4891 karakter per permintaan)."""
    if len(content) > 4891:
        return 'Masukan tidak boleh melebihi 4891 karakter!'
    salt = str(random.randint(0, 50))
    # Daftar kunci di: http://api.fanyi.baidu.com/api/trans/product/desktop
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    params = {
        'q': content,
        'from': t_from,
        'to': t_to,
        'appid': appid,
        'salt': salt,
        'sign': sign,
    }
    response = requests.get(
        'http://api.fanyi.baidu.com/api/trans/vip/translate',
        params=params,
    )
    res = response.json()['trans_result'][0]['dst']
    return res


if __name__ == '__main__':
    # Ganti dengan APP ID dan secretKey Anda
    APP_ID = 'xxxx'
    SECRET_KEY = 'xxxxx'

    with open('q_gov.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        result = []
        for line in tqdm(lines[:100]):
            line = line.strip()
            if not line:
                continue
            ori_question, ans = line.split('||')
            # Back-translation: zh→en→zh (ubah ke id/en untuk bahasa Indonesia)
            temp = baidu_translate(
                content=ori_question,
                appid=APP_ID,
                secretKey=SECRET_KEY,
                t_from='zh',
                t_to='en',
            )
            trans_question = baidu_translate(
                content=temp,
                appid=APP_ID,
                secretKey=SECRET_KEY,
                t_from='en',
                t_to='zh',
            )
            res = '||'.join([ori_question, trans_question, ans])
            result.append(res)

    with open('q_gov_aug.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(result))

    print('Augmentasi selesai. Hasil disimpan ke q_gov_aug.txt')

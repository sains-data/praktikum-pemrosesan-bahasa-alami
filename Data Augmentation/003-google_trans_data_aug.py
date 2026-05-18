# -*- coding: utf-8 -*-
"""
Augmentasi batch memakai endpoint web Google Translate (bukan API resmi).
Gabung banyak kalimat dengan newline untuk mempercepat terjemahan massal.
"""
import time
import urllib.parse
import urllib.request

import execjs
from tqdm import tqdm as tqdm


def open_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'
    }
    req = urllib.request.Request(url=url, headers=headers)
    response = urllib.request.urlopen(req)
    data = response.read().decode('utf-8')
    return data


def translate(content, tk, sl='en', tl='zh-CN'):
    """
    Memanggil Google Translate web.
    sl: bahasa sumber, tl: bahasa target (sesuaikan kebutuhan back-translation).
    """
    if len(content) > 4891:
        print("Teks terjemahan melebihi batas panjang!")
        return None
    content = urllib.parse.quote(content)
    url = (
        "http://translate.google.cn/translate_a/single?client=t"
        f"&sl={sl}&tl={tl}&hl={tl}&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca"
        "&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1"
        "&srcrom=0&ssel=0&tsel=0&kc=2&tk=%s&q=%s" % (tk, content)
    )
    result = open_url(url)
    return result


class Kaihua:
    """Menghitung token tk yang diperlukan Google Translate (via JavaScript)."""

    def __init__(self):
        self.ctx = execjs.compile("""
        function TL(a) {
        var k = "";
        var b = 406644;
        var b1 = 3293161072;
        var jd = ".";
        var $b = "+-a^+6";
        var Zb = "+-3^+b+-f";
        for (var e = [], f = 0, g = 0; g < a.length; g++) {
            var m = a.charCodeAt(g);
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
            e[f++] = m >> 18 | 240,
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
            e[f++] = m >> 6 & 63 | 128),
            e[f++] = m & 63 | 128)
        }
        a = b;
        for (f = 0; f < e.length; f++) a += e[f],
        a = RL(a, $b);
        a = RL(a, Zb);
        a ^= b1 || 0;
        0 > a && (a = (a & 2147483647) + 2147483648);
        a %= 1E6;
        return a.toString() + jd + (a ^ b)
        };
        function RL(a, b) {
        var t = "a";
        var Yb = "+";
        for (var c = 0; c < b.length - 2; c += 3) {
            var d = b.charAt(c + 2),
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
        }
        return a
        }
        """)

    def getTk(self, text):
        return self.ctx.call("TL", text)


if __name__ == '__main__':
    js = Kaihua()
    success = 0

    # Daftar kalimat sumber — ganti dengan muatan dari file Anda
    a_list = [
        'what is your name?',
        'what are you doing?',
    ]

    a_trans_list = []

    """
    Alur batch:
    - Gabungkan combined_length kalimat dengan pemisah newline.
    - Terjemahkan sekali, lalu pecah kembali ke daftar kalimat.
    - Untuk ~100k kalimat pendek, perkiraan waktu ~20 menit (bergantung jaringan).
    - Total karakter per batch sebaiknya < 5000 (batasan Google).
    """
    combined_length = 100

    for i in tqdm(range(len(a_list) // combined_length + 1)):
        content = '\n'.join(a_list[i * combined_length:(i + 1) * combined_length])
        if content == '':
            continue
        get_trans = True
        while get_trans:
            try:
                tk = js.getTk(content)
                result = translate(content, tk, sl='en', tl='zh-CN')
                result = result.replace('null', 'None')
                result = result.replace('true', 'True')
                result = result.replace('false', 'False')
                result = eval(result)
                trans_result = ''
                for item in result[0][0:-1]:
                    trans_result += item[0]
                train_result = trans_result.split('\n')
                n_src = len(content.split('\n'))
                print(len(train_result), n_src)
                if len(train_result) != n_src:
                    print('Jumlah baris tidak cocok, coba ulang...')
                    raise Exception('alignment')
                a_trans_list.extend(train_result)
                success += 1
                get_trans = False
                time.sleep(0.1)
            except Exception:
                pass
        if success % 10 == 0 and success > 0:
            print('Batch berhasil:', success)
            print('Contoh terjemahan:', a_trans_list[-10:])
        time.sleep(0.1)

    assert len(a_list) == len(a_trans_list)
    print('Terjemahan batch selesai!')
    print(a_trans_list)

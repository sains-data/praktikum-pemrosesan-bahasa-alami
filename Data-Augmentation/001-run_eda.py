#!/usr/bin/python
# pip install nlpcda
# pip install nlpcda -i https://pypi.douban.com/simple/  # alternatif mirror jika perlu
from nlpcda import Randomword
from nlpcda import Similarword
from nlpcda import Homophone
from nlpcda import RandomDeleteChar
from nlpcda import Ner
from nlpcda import CharPositionExchange
from nlpcda import baidu_translate
from nlpcda import EquivalentChar


def test_EquivalentChar(test_str, create_num=2, change_rate=0.5):
    """Penggantian karakter/setara (mis. varian angka atau huruf mirip)."""
    s = EquivalentChar(create_num=create_num, change_rate=change_rate)
    return s.replace(test_str)


def test_Randomword(test_str, create_num=2, change_rate=0.2):
    """Penggantian entitas acak (mis. nama perusahaan dari extdata/company.txt)."""
    smw = Randomword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Similarword(test_str, create_num=2, change_rate=0.2):
    """Penggantian kata dengan sinonim secara acak."""
    smw = Similarword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Homophone(test_str, create_num=2, change_rate=0.2):
    """Penggantian dengan kata homofon / bunyi mirip."""
    hoe = Homophone(create_num=create_num, change_rate=change_rate)
    return hoe.replace(test_str)


def test_RandomDeleteChar(test_str, create_num=2, change_rate=0.1):
    """Menghapus karakter secara acak."""
    smw = RandomDeleteChar(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_CharPositionExchange(test_str, create_num=2, change_rate=0.5):
    """Menukar posisi karakter/kata."""
    smw = CharPositionExchange(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_baidu_translate(test_str):
    """Back-translation via Baidu: zh→en→zh (sesuaikan bahasa untuk Indonesia)."""
    # Daftar APP ID dan secretKey di http://api.fanyi.baidu.com/
    temp = baidu_translate(content=test_str, appid='XXXX', secretKey='XXXX', t_from='zh', t_to='en')
    res = baidu_translate(content=temp, appid='XXXX', secretKey='XXXX', t_from='en', t_to='zh')
    return res


if __name__ == '__main__':
    ts = (
        'Ini contoh teks untuk augmentasi data NLP. '
        'Cuaca hari ini cerah dan nyaman; paket nlpcda memudahkan augmentasi '
        'untuk meningkatkan generalisasi model.'
    )

    rs1 = test_EquivalentChar(ts)
    print('*' * 10 + ' Penggantian karakter setara ' + '*' * 10)
    for _ in rs1:
        print(_)

    rs2 = test_Randomword(ts)
    print('*' * 10 + ' Penggantian entitas acak ' + '*' * 10)
    for _ in rs2:
        print(_)

    rs3 = test_Similarword(ts)
    print('*' * 10 + ' Penggantian sinonim ' + '*' * 10)
    for _ in rs3:
        print(_)

    rs4 = test_Homophone(ts)
    print('*' * 10 + ' Penggantian homofon ' + '*' * 10)
    for _ in rs4:
        print(_)

    rs5 = test_RandomDeleteChar(ts)
    print('*' * 10 + ' Penghapusan karakter acak ' + '*' * 10)
    for _ in rs5:
        print(_)

    rs6 = test_CharPositionExchange(ts)
    print('*' * 10 + ' Penukaran posisi ' + '*' * 10)
    for _ in rs6:
        print(_)

    print('*' * 10 + ' Augmentasi terjemahan (back-translation) ' + '*' * 10)
    print('Asli:', ts)
    print('Hasil:', test_baidu_translate(ts))

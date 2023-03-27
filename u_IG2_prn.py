#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

import ctypes
from copy import copy



sIG2File_SIGNATURE = b'\x03IG2\xffDATA\xeeFILE\xddVERSION\xcc2\xbb\xaa\x99\x88wf'


class TStationCoords(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        # Широта и Долгота (в сотых градуса, отрицат. знач-я для - юж.ш., зап.д.)
        ('Latitude', ctypes.c_int16),
        ('Longitude', ctypes.c_int16),
        # crdFlagNo=0; crdFlagUnknown=1; crdFlagGPS=2; crdFlagAbout=3
        ('Flags', ctypes.c_uint8),
        # ... особенно актуально для точно НЕ известных станций
        ('StationShortName', ctypes.c_char * 9)
    ]


# заголовок ионограмм формата IG2 (авторы формата - А.Егошин, В.Батухтин)
class TIG2FileHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('Signature', ctypes.c_uint8 * 30),
        # (устар.) в каком режиме создан файл',  HFS или SA 
        ('wFileType', ctypes.c_uint16),
        # дата
        ('wDay', ctypes.c_uint16),
        ('wMonth', ctypes.c_uint16),
        ('wYear', ctypes.c_uint16),
        # время
        ('wHour', ctypes.c_uint16),
        ('wMinute', ctypes.c_uint16),
        ('wSecond', ctypes.c_uint16),
        # число БПФ, тип окна
        ('wNFFT', ctypes.c_uint16),
        ('wFFTWindType', ctypes.c_uint16),
        # трасса (устар., но м.б. актуальна при работе со старыми данными) 
        ('wTraceID', ctypes.c_uint16),
        # диапазон сигнала
        ('wADCType', ctypes.c_uint16),
        # (устар.) число отсчетов в файле 
        ('lSampleNumber',  ctypes.c_int32), 
        # частота АЦП, скорость и диапазон зондирования 
        ('exADCFreq', ctypes.c_double),
        ('exSoundSpeed', ctypes.c_double),
        ('exStartFreq', ctypes.c_double),
        ('exStopFreq', ctypes.c_double),
        # текстовая информация
        ('sInfo', ctypes.c_uint8 * 30),
        # (устар.) корректность данных 
        ('boIsBadFile',  ctypes.c_uint8),      
        # размер спектра
        ('wIGSpecSize',  ctypes.c_uint16),
        # (устар.) вид связи
        ('iConnectType',  ctypes.c_int16),
        # нижняя граница разностной частоты 
        ('iLowestDiffFreq', ctypes.c_int16),
        # параметры спектрального усреднения по Уэлчу (неисп.?) 
        ('wOverlapCoeff',  ctypes.c_uint16),
        ('wSpecNumber',  ctypes.c_uint16),
        ('wSampleInSegm',  ctypes.c_uint16),
        ('wBufferSize',  ctypes.c_uint16), 
        # диапазон временной задержки 
        ('rDelayRange',  ctypes.c_float),
        # ширина спектра в ионограмме /0 = 100 кГц/
        ('exIGSpecWidth',  ctypes.c_double),
        # (устар.) размер элемента в битах',  <12> или 16 - имеется ввиду отсчеты временного ряда, а не ионограммы 
        ('wSampleSizeType',  ctypes.c_uint16),
        # БПФ с плавающей точкой, иначе с <фиксированной> 
        ('boIsFFTFloat', ctypes.c_uint8), 
        # Задержка - параметр для синтезатора [мс]
        ('SynthDelay', ctypes.c_double),
        # координаты приемника
        ('ReceiverCoords', TStationCoords),
        # координаты передатчика
        ('TransmitterCoords', TStationCoords),
        # Смещение разностной частоты, внесенное в синтезатор [Гц]
        ('SynthDifFreqToSub', ctypes.c_uint16),
        #
        ('Reserved', ctypes.c_uint8 * 14)
    ]



def loadIG2(fn):
    data = open(fn, 'rb').read()

    if data[0:30] == sIG2File_SIGNATURE:

        I = ctypes.cast(data[0:200], ctypes.POINTER(TIG2FileHeader))
        
        if ((I.contents.rDelayRange < 1e-6)|(I.contents.rDelayRange > 1e3)):
            rDelayRange = I.contents.exADCFreq / 2 / I.contents.exSoundSpeed
            SpecElemCount = int(rDelayRange * I.contents.exSoundSpeed * I.contents.wNFFT / I.contents.exADCFreq)  #
        else:
            SpecElemCount = int(I.contents.rDelayRange * I.contents.exSoundSpeed * I.contents.wNFFT / I.contents.exADCFreq)  #
        
        if ((I.contents.exIGSpecWidth < 1e-6)|(I.contents.exIGSpecWidth > 1e3)):
            SpecCount = int((I.contents.exStopFreq - I.contents.exStartFreq) * 1000 / 100)
        else:
            SpecCount = int((I.contents.exStopFreq - I.contents.exStartFreq) * 1000 / I.contents.exIGSpecWidth)
        
        return copy(data[200:]), SpecCount, SpecElemCount, copy(I.contents)

    else:
        return 0,0,0,0


def getTextInfoByIG2Header(h):
    return "{}.{:0>2d}.{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}".format(h.wYear, h.wMonth, h.wDay, h.wHour, h.wMinute, h.wSecond)  \
           + chr(10) + "{}–{} MHz, {} kHz/s".format(h.exStartFreq, h.exStopFreq, h.exSoundSpeed) \
           + chr(10) + "{} {}".format(h.rDelayRange, h.exIGSpecWidth)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]

        Arr, SpecCount, SpecElemCount, Hdr = loadIG2(fn)

        if (SpecCount > 0) & (SpecElemCount > 0):
            #print( getTextInfoByIG2Header(Hdr) )

            import numpy as np
            a = np.frombuffer(Arr, dtype=np.uint8)
            print(len(a))
            print(SpecCount*SpecElemCount)

            fp = open(fn+'-out.txt', 'w')
            for i in range(len(a)):
                fp.write( str(a[i]) + chr(10) )
            fp.close()

        else: 
            print("Unknown format")


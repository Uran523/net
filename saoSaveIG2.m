%   Функция записи данных в файл формата IG2
%   
%   Использование:
%   
%   saoSaveIG2( filename, Img, Hdr );
%   
%   Img - это матрица чисел double со значениями [0, 255]
%   (должна быть преобразована к этому диапазону)
%   Hdr - заголовок ионограммы с различными полями (см.функцию saoLoadIG2)
%

function saoSaveIG2( filename, Img, Hdr )

NumOfSpecs = size(Img, 1);
SpecSize = size(Img, 2);

fid = fopen(filename, 'wb');

ig2sgn = [3, 73, 71, 50, 255, 68, 65, 84, 65, 238, 70, 73, 76, 69, 221, 86, 69, 82, 83, 73, 79, 78, 204, 50, 187, 170, 153, 136, 119, 102];

fwrite(fid, ig2sgn);

fwrite(fid, Hdr.wFileType , 'uint16');   
fwrite(fid, Hdr.wDay      , 'uint16');   
fwrite(fid, Hdr.wMonth    , 'uint16');   
fwrite(fid, Hdr.wYear     , 'uint16');   
fwrite(fid, Hdr.wHour     , 'uint16');   
fwrite(fid, Hdr.wMinute   , 'uint16');   
fwrite(fid, Hdr.wSecond   , 'uint16');   
fwrite(fid, Hdr.wNFFT     , 'uint16');   
fwrite(fid, Hdr.wFFTWindType, 'uint16'); 
fwrite(fid, Hdr.wTraceID, 'uint16');     
fwrite(fid, Hdr.wADCType, 'uint16');     
fwrite(fid, Hdr.lSampleNumber, 'int32'); 
fwrite(fid, Hdr.exADCFreq, 'double');
fwrite(fid, Hdr.exSoundSpeed, 'double');
fwrite(fid, Hdr.exStartFreq, 'double');
fwrite(fid, Hdr.exStopFreq, 'double');

fwrite(fid, Hdr.sSAID);

fwrite(fid, Hdr.boIsBadFile, 'uint8');
fwrite(fid, Hdr.wIGSpecSize, 'uint16');
fwrite(fid, Hdr.iConnectType, 'int16');
fwrite(fid, Hdr.iLowestDiffFreq, 'int16');
fwrite(fid, Hdr.wOverlapCoeff, 'uint16');
fwrite(fid, Hdr.wSpecNumber, 'uint16');
fwrite(fid, Hdr.wSampleInSegm, 'uint16');
fwrite(fid, Hdr.wBufferSize, 'uint16');

fwrite(fid, Hdr.rDelayRange, 'single');
fwrite(fid, Hdr.boIsFFTFloat, 'uint8');    
fwrite(fid, Hdr.exIGSpecWidth, 'float64');
fwrite(fid, Hdr.wSampleSizeType, 'uint16');

% заполнение нулями остатка заголовка
fwrite(fid, 0, 'int8', 200-ftell(fid)-1);

fwrite(fid, cast(Img,'uint8'), 'uint8');
    
fclose(fid);

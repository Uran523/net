%   Функция загрузки данных из файла формата IG2
%   
%   Использование:
%   
%   [Img, NumOfSpecs, SpecSize, Hdr] = saoLoadIG2( filename );
%   
%   Img - это матрица чисел double со значениями [0, 255] (так как преобразована из uint8)
%   NumOfSpecs - число спектров
%   SpecSize   - размер спектра
%   Hdr - заголовок ионограммы с различными полями (см.функцию)
%


function [Img, NumOfSpecs, SpecSize, Hdr] = saoLoadIG2( filename )

fid = fopen(filename, 'r');
% проверяется только сигнатура файла, корректность размеров не проверяется
sgn0 = fread(fid, 4, 'uint8=>char');
if (sgn0(1)==3) && (sgn0(2)=='I') && (sgn0(3)=='G') && (sgn0(4)=='2')
    % sgn begin - Ok

    fseek(fid, 30, 'bof'); % skip signature rest
    Hdr.wFileType  = fread(fid, 1, 'uint16');   %  в каком режиме создан файл: HFS или SA
    Hdr.wDay       = fread(fid, 1, 'uint16');   %  дата 
    Hdr.wMonth     = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wYear      = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wHour      = fread(fid, 1, 'uint16');   %  время
    Hdr.wMinute    = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wSecond    = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wNFFT      = fread(fid, 1, 'uint16');   %  число БПФ              
    Hdr.wFFTWindType = fread(fid, 1, 'uint16'); %  тип окна               
    Hdr.wTraceID = fread(fid, 1, 'uint16');     %  трасса                 
    Hdr.wADCType = fread(fid, 1, 'uint16');     %  диапазон сигнала       
    Hdr.lSampleNumber = fread(fid, 1, 'int32'); %  число отсчетов в файле (не в этом, а в SEQ)

    % { частота АЦП, скорость и диапазон зондирования }
    Hdr.exADCFreq = fread(fid, 1, 'double');
    Hdr.exSoundSpeed = fread(fid, 1, 'double');
    Hdr.exStartFreq = fread(fid, 1, 'double');
    Hdr.exStopFreq = fread(fid, 1, 'double');

    Hdr.sSAID = fread(fid, 30, 'uint8=>char'); % дополнительная текстовая информация
    Hdr.boIsBadFile = fread(fid, 1, 'uint8');
    Hdr.wIGSpecSize = fread(fid, 1, 'uint16');
    Hdr.iConnectType = fread(fid, 1, 'int16');
    Hdr.iLowestDiffFreq = fread(fid, 1, 'int16'); % нижняя граница разностной частоты

    % параметры спекрального усреднения по Уэлчу
    Hdr.wOverlapCoeff = fread(fid, 1, 'uint16');
    Hdr.wSpecNumber = fread(fid, 1, 'uint16');
    Hdr.wSampleInSegm = fread(fid, 1, 'uint16');
    Hdr.wBufferSize = fread(fid, 1, 'uint16');

    Hdr.rDelayRange = fread(fid, 1, 'single'); % диапазон временной задержки
    Hdr.boIsFFTFloat = fread(fid, 1, 'uint8');  % БПФ с плавающей точкой, иначе с <фиксированной>
    Hdr.exIGSpecWidth = fread(fid, 1, 'double'); % ширина спекра в ионограмме /0 = 100 кГц/
    Hdr.wSampleSizeType = fread(fid, 1, 'uint16'); % размер элемента в битах: <12> или 16


    if (Hdr.rDelayRange < 1e-6)
        Hdr.rDelayRange = Hdr.exADCFreq / 2 / Hdr.exSoundSpeed;
    end

    if (Hdr.exIGSpecWidth < 1e-6)
        Hdr.exIGSpecWidth = 100;
    end

    NumOfSpecs = round( (Hdr.exStopFreq - Hdr.exStartFreq) * 1000 / Hdr.exIGSpecWidth );
    SpecSize = round( Hdr.rDelayRange * Hdr.exSoundSpeed * Hdr.wNFFT / Hdr.exADCFreq );

    fseek(fid, 200, 'bof');
    img_l = fread(fid, NumOfSpecs*SpecSize, 'uint8');

    Img = reshape(img_l, NumOfSpecs, SpecSize);

else
    % ... bad file
    Img = 0;
    NumOfSpecs = 0;
    SpecSize = 0;
    Hdr = 0;
end
fclose(fid);

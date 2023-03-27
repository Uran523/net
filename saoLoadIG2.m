%   ������� �������� ������ �� ����� ������� IG2
%   
%   �������������:
%   
%   [Img, NumOfSpecs, SpecSize, Hdr] = saoLoadIG2( filename );
%   
%   Img - ��� ������� ����� double �� ���������� [0, 255] (��� ��� ������������� �� uint8)
%   NumOfSpecs - ����� ��������
%   SpecSize   - ������ �������
%   Hdr - ��������� ���������� � ���������� ������ (��.�������)
%


function [Img, NumOfSpecs, SpecSize, Hdr] = saoLoadIG2( filename )

fid = fopen(filename, 'r');
% ����������� ������ ��������� �����, ������������ �������� �� �����������
sgn0 = fread(fid, 4, 'uint8=>char');
if (sgn0(1)==3) && (sgn0(2)=='I') && (sgn0(3)=='G') && (sgn0(4)=='2')
    % sgn begin - Ok

    fseek(fid, 30, 'bof'); % skip signature rest
    Hdr.wFileType  = fread(fid, 1, 'uint16');   %  � ����� ������ ������ ����: HFS ��� SA
    Hdr.wDay       = fread(fid, 1, 'uint16');   %  ���� 
    Hdr.wMonth     = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wYear      = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wHour      = fread(fid, 1, 'uint16');   %  �����
    Hdr.wMinute    = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wSecond    = fread(fid, 1, 'uint16');   %  -//-
    Hdr.wNFFT      = fread(fid, 1, 'uint16');   %  ����� ���              
    Hdr.wFFTWindType = fread(fid, 1, 'uint16'); %  ��� ����               
    Hdr.wTraceID = fread(fid, 1, 'uint16');     %  ������                 
    Hdr.wADCType = fread(fid, 1, 'uint16');     %  �������� �������       
    Hdr.lSampleNumber = fread(fid, 1, 'int32'); %  ����� �������� � ����� (�� � ����, � � SEQ)

    % { ������� ���, �������� � �������� ������������ }
    Hdr.exADCFreq = fread(fid, 1, 'double');
    Hdr.exSoundSpeed = fread(fid, 1, 'double');
    Hdr.exStartFreq = fread(fid, 1, 'double');
    Hdr.exStopFreq = fread(fid, 1, 'double');

    Hdr.sSAID = fread(fid, 30, 'uint8=>char'); % �������������� ��������� ����������
    Hdr.boIsBadFile = fread(fid, 1, 'uint8');
    Hdr.wIGSpecSize = fread(fid, 1, 'uint16');
    Hdr.iConnectType = fread(fid, 1, 'int16');
    Hdr.iLowestDiffFreq = fread(fid, 1, 'int16'); % ������ ������� ���������� �������

    % ��������� ������������ ���������� �� �����
    Hdr.wOverlapCoeff = fread(fid, 1, 'uint16');
    Hdr.wSpecNumber = fread(fid, 1, 'uint16');
    Hdr.wSampleInSegm = fread(fid, 1, 'uint16');
    Hdr.wBufferSize = fread(fid, 1, 'uint16');

    Hdr.rDelayRange = fread(fid, 1, 'single'); % �������� ��������� ��������
    Hdr.boIsFFTFloat = fread(fid, 1, 'uint8');  % ��� � ��������� ������, ����� � <�������������>
    Hdr.exIGSpecWidth = fread(fid, 1, 'double'); % ������ ������ � ���������� /0 = 100 ���/
    Hdr.wSampleSizeType = fread(fid, 1, 'uint16'); % ������ �������� � �����: <12> ��� 16


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

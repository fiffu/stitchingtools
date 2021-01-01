@ECHO off
SETLOCAL enabledelayedexpansion

REM ### SETTINGS ###
SET fileFormat=.mkv
SET mapStreams=-map 0:0 -map 0:2 -map 0:4 -map 0:5
SET outputFolder=done
REM ################


IF NOT EXIST %outputFolder% MKDIR %outputFolder%

FOR %%k IN (*) DO (
    SET file="%%k"
    SET outFile="%outputFolder%/%%k"

    REM Get tail end of `filename` string then try replacing the `extension`.
    REM If outcome differs, `extension` is a substr of `filename`.
    REM ,-1 excludes the trailing " in !file!
    SET tail=!file:~-10,-1!
    SET tail_=!tail:%fileFormat%=!

    REM Should we process this file?
    SET processFile=true
    IF /i !tail!==!tail_! (
        SET processFile=false
    )
    IF EXIST !outFile! (
        SET processFile=false
    )

    REM Pass options to ffmpeg
    IF !processFile!==true (
        ffmpeg -i !file! -c copy %mapStreams% !outFile! -y
    )
)

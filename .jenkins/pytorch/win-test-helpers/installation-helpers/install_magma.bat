REM if "%CUDA_VERSION%" == "9" set CUDA_SUFFIX=cuda92
REM if "%CUDA_VERSION%" == "10" set CUDA_SUFFIX=cuda101

REM if "%CUDA_SUFFIX%" == "" (
REM   echo unknown CUDA version, please set `CUDA_VERSION` to 9 or 10.
REM   exit /b 1
REM )

REM if "%REBUILD%"=="" (
REM   if "%BUILD_ENVIRONMENT%"=="" (
REM     curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/magma_2.5.1_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --output %TMP_DIR_WIN%\magma_2.5.1_%CUDA_SUFFIX%_%BUILD_TYPE%.7z
REM   ) else (
REM     aws s3 cp s3://ossci-windows/magma_2.5.1_%CUDA_SUFFIX%_%BUILD_TYPE%.7z %TMP_DIR_WIN%\magma_2.5.1_%CUDA_SUFFIX%_%BUILD_TYPE%.7z --quiet
REM   )
REM   7z x -aoa %TMP_DIR_WIN%\magma_2.5.1_%CUDA_SUFFIX%_%BUILD_TYPE%.7z -o%TMP_DIR_WIN%\magma
REM )
REM set MAGMA_HOME=%TMP_DIR_WIN%\magma

import os
from os.path import exists
import subprocess
import sys
import contextlib


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


build_environment = os.environ['BUILD_ENVIRONMENT']
tmp_dir = os.environ['TMP_DIR']


if exists(tmp_dir + '/ci_scripts/pytorch_env_restore.bat'):

    subprocess.call(tmp_dir + '/ci_scripts/pytorch_env_restore.bat', shell=True)
    sys.exit(0)


os.environ['PATH'] = 'C:\\Program Files\\CMake\\bin;C:\\Program Files\\7-Zip;C:\\'
 'ProgramData\\chocolatey\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Amazon\\'
  'AWSCLI;C:\\Program Files\\Amazon\\AWSCLI\\bin;' + os.environ['PATH']

# Install Miniconda3
os.environ['INSTALLER_DIR'] = os.environ['SCRIPT_HELPERS_DIR'] + '\\installation-helpers'

# Miniconda has been installed as part of the Windows AMI with all the dependencies.
# We just need to activate it here
try:
    subprocess.call(os.environ['INSTALLER_DIR'] + '\\activate_miniconda3.py', shell=True)
except Exception as e:

    subprocess.run(['echo', 'activate conda failed'])
    subprocess.run(['echo', e])
    sys.exit()

# extra conda dependencies for testing purposes
if not build_environment == '':

    try:
        subprocess.run(['conda', 'install', '-y', '-q', 'mkl', 'protobuf',\
         'numba', 'scipy=1.6.2', 'typing_extensions', 'dataclasses'])
    except Exception as e:

        subprocess.run(['echo', 'conda install failed'])
        subprocess.run(['echo', e])
        sys.exit()


with pushd('.'):

    try:
        if os.environ['VC_VERSION'] == '':
            subprocess.run(['C:\\Program Files (x86)\\Microsoft Visual Studio\\' +\
            os.environ['VC_YEAT'] + '\\' + os.environ['VC_VERSION'] +\
             '\\VC\Auxiliary\Build\vcvarsall.bat', 'x64'])

        else:
            subprocess.run(['C:\\Program Files (x86)\\Microsoft Visual Studio\\' +\
            os.environ['VC_YEAT'] + '\\' + os.environ['VC_VERSION'] +\
             '\\VC\Auxiliary\Build\vcvarsall.bat', 'x64', '-vcvars_ver=' + os.environ['VC_VERSION']])

        subprocess.run(['@echo', 'on'])

    except Exception as e:

        subprocess.run(['echo', 'vcvarsall failed'])
        subprocess.run(['echo', e])
        sys.exit()


# The version is fixed to avoid flakiness: https://github.com/pytorch/pytorch/issues/31136
# =======
# Pin unittest-xml-reporting to freeze printing test summary logic, related: https://github.com/pytorch/pytorch/issues/69014

try:
    subprocess.run(['pip', 'install', "ninja==1.10.0.post1", 'future',\
     "hypothesis==5.35.1", "expecttest==0.1.3", "librosa>=0.6.2", "scipy==1.6.3",\
      'psutil', 'pillow', "unittest-xml-reporting<=3.2.0,>=2.0.0", 'pytest',\
       'pytest-xdist', 'pytest-shard', 'pytest-rerunfailures', 'sympy',\
        "xdoctest==1.0.2", "pygments==2.12.0", "opt-einsum>=3.3"])

except Exception as e:

    subprocess.run(['echo', 'install dependencies failed'])
    subprocess.run(['echo', e])
    sys.exit()


os.environ['DISTUTILS_USE_SDK'] = 1

if os.environ['USE_CUDA'] == '1':

    os.environ['CUDA_PATH']='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v'\
     + str(os.environ['CUDA_VERSION'])

    # version transformer, for example 10.1 to 10_1.
    os.environ['VERSION_SUFFIX']=str(os.environ['CUDA_VERSION']).replace('.','_')
    os.environ['CUDA_PATH_V' + str(os.environ['VERSION_SUFFIX'])]=str(os.environ['CUDA_PATH'])

    os.environ['CUDNN_LIB_DIR']=str(os.environ['CUDA_PATH']) + '\\lib\\x64'
    os.environ['CUDA_TOOLKIT_ROOT_DIR']=str(os.environ['CUDA_PATH'])
    os.environ['CUDNN_ROOT_DIR']=str(os.environ['CUDA_PATH'])
    os.environ['NVTOOLSEXT_PATH']='C:\\Program Files\\NVIDIA Corporation\\NvToolsExt'
    os.environ['PATH']=str(os.environ['CUDA_PATH']) + '\\bin;' + str(os.environ['CUDA_PATH'])+\
    '\\libnvvp;' + str(os.environ['PATH'])
    os.environ['NUMBAPRO_CUDALIB']=str(os.environ['CUDA_PATH']) + '\\bin'
    os.environ['NUMBAPRO_LIBDEVICE']=str(os.environ['CUDA_PATH']) + '\\nvvm\\libdevice'
    os.environ['NUMBAPRO_NVVM']=str(os.environ['CUDA_PATH']) + '\\nvvm\\bin\\nvvm64_32_0.dll'


os.environ['PYTHONPATH'] = str(os.environ['TMP_DIR_WIN']) + '\\build;' + str(os.environ['PYTHONPATH'])

if not str(os.environ['BUILD_ENVIRONMENT']) == '':

    with pushd(str(os.environ['TMP_DIR_WIN']) + '\\build'):

        subprocess.run(['copy', '/Y', str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN'])+\
        '\\' + str(os.environ['IMAGE_COMMIT_TAG']) + '.7z', str(os.environ['TMP_DIR_WIN']) + '\\'])

        # 7z: -aos skips if exists because this .bat can be called multiple times

        subprocess.run(['7z', 'x', str(os.environ['TMP_DIR_WIN']) + '\\' +\
         str(os.environ['IMAGE_COMMIT_TAG']) + '.7z',  '-aos'])

else:

    subprocess.run(['xcopy', '/s', str(os.environ['CONDA_PARENT_DIR']) + \
    '\\Miniconda3\\Lib\\site-packages\\torch', str(os.environ['TMP_DIR_WIN']) +\
    '\\build\\torch\\'])

subprocess.run(['@echo', 'off'])
subprocess.run(['echo', '\@echo', 'off', '>>', str(os.environ['TMP_DIR_WIN']) +\
'/ci_scripts/pytorch_env_restore.bat'])

restore_file = open(str(os.environ['TMP_DIR_WIN']) + '/ci_scripts/pytorch_env_restore.bat', 'a+')
set_file = open('set', 'r')
restore_file.write(set_file.read())
restore_file.close()
set_file.close()

subprocess.run(['@echo', 'on'])

if not str(os.environ['BUILD_ENVIRONMENT']) == '':

    # Create a shortcut to restore pytorch environment
    subprocess.run(['echo', '\@echo', 'off', '>>', str(os.environ['TMP_DIR_WIN']) +\
    '/ci_scripts/pytorch_env_restore_helper.bat'])
    subprocess.run(['echo', 'call', str(os.environ['TMP_DIR_WIN']) + '/ci_scripts/pytorch_env_restore.bat',\
     '>>', str(os.environ['TMP_DIR_WIN']) + '/ci_scripts/pytorch_env_restore_helper.bat'])
    subprocess.run(['echo', 'cd', '/D', str(os.environ['CD']), '>>', str(os.environ['TMP_DIR_WIN']) +\
    '/ci_scripts/pytorch_env_restore_helper.bat'])

    subprocess.run(['aws', 's3', 'cp', '"s3://ossci-windows/Restore PyTorch Environment.lnk"',\
     '"C:\\Users\\circleci\\Desktop\\Restore PyTorch Environment.lnk"'])

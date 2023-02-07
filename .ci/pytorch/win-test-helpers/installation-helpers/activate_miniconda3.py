import os
import subprocess
import sys
import pathlib


# Be conservative here when rolling out the new AMI with conda. This will try
# to install conda as before if it couldn't find the conda installation. This
# can be removed eventually after we gain enough confidence in the AMI


conda_parent_dir = os.environ['CONDA_PARENT_DIR']
tmp_dir_win = os.environ['TMP_DIR_WIN']


if install_fresh_conda == '1':

    try:
        result = subprocess.run('echo Installing conda to: ' + conda_parent_dir + '\\Miniconda3', shell=True)
        result.check_returncode()

        result = subprocess.run('curl --retry 3 -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe ' +
                '--output ' + tmp_dir_win + '\\Miniconda3-latest-Windows-x86_64.exe', shell=True)
        result.check_returncode()

        subprocess.run('ls ' + tmp_dir_win, shell=True)
        subprocess.run('ls C:\\Jenkins\\Miniconda3', shell=True)
        subprocess.run('echo ' + str(os.environ['TMP_DIR_WIN']), shell=True)
        subprocess.run('echo ' + tmp_dir_win + '\\Miniconda3-latest-Windows-x86_64.exe', shell=True)
        subprocess.run('echo ' + str(result.stderr), shell=True)
        subprocess.run('echo ' + str(result.stdout), shell=True)

        os.system(str(pathlib.Path(__file__).parent.resolve()) + '\\conda_test.bat ' + tmp_dir_win + ' ' + conda_parent_dir)

        '''
        subprocess.Popen('Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe ' +
            '/RegisterPython=0 /S /AddToPath=0 /D=' + conda_parent_dir + '\\Miniconda3', cwd=tmp_dir_win)


        # run gave errors with conda here, error may be related to installer bug
        result = subprocess.run(str(tmp_dir_win + '\\Miniconda3-latest-Windows-x86_64.exe ' +
            '/InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=' + conda_parent_dir + '\\Miniconda3'), shell=True)
        subprocess.run('echo ' + str(result.stderr), shell=True)
        subprocess.run('echo ' + str(result.stdout), shell=True)
        result.check_returncode()
        '''

        result = subprocess.run('echo Installed conda to: ' + conda_parent_dir + '\\Miniconda3', shell=True)
        result.check_returncode()


        os.environ['PATH'] = conda_parent_dir + '\\Miniconda3\\Library\\bin;' + conda_parent_dir +\
            '\\Miniconda3;' + conda_parent_dir + '\\Miniconda3\\Scripts;' + os.environ['PATH']

    except Exception as e:

        subprocess.run('ls ' + tmp_dir_win, shell=True)
        subprocess.run('ls C:\\Jenkins\\Miniconda3', shell=True)

        subprocess.run('echo activate conda failed', shell=True)
        subprocess.run('echo ' + str(e), shell=True)
        subprocess.run('ls C:\\Jenkins\\Miniconda3', shell=True)
        sys.exit()


# Activate conda so that we can use its commands, i.e. conda, python, pip
subprocess.run('conda create --prefix ' + conda_parent_dir + '\\Miniconda3\\\envs\\test_env', shell=True)


if install_fresh_conda == '1':

    try:

        result = subprocess.run('conda install -n test_env -y -q numpy cffi pyyaml boto3 libuv', shell=True)
        result.check_returncode()

        result = subprocess.run('conda install -n test_env -y -q -c conda-forge cmake=3.22.3', shell=True)
        result.check_returncode()

    except Exception as e:

        subprocess.run('echo activate conda failed', shell=True)
        subprocess.run('echo ' + str(e), shell=True)
        sys.exit()

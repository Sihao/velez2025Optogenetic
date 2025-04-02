ANTsX Environment Setup Notes

Holland Brown

Updated 2023-05-15
Created 2023-05-15

Installed in conda env zfbrain:
    - ANTs
    - numpy
    - glob2

---------------------------------------------------------------------------

1. ANTs configuration

    (A) System Requirements
        >> CMake
            - installed software in /Applications
            - configured commandline tools in base env

        >> Installed uncrustify, cppcheck, ccache in base env using homebrew

        >> XCode app & commandline developer tools
            - installed commandline developer tools in base env

    (B) Compiled ANTs with shared libraries

    export ANTSPATH=/opt/ANTs/bin
    export PATH=${ANTSPATH}:$PATH

        >> Note: using above export cmds sets env vbls for current terminal; reverts when new terminal is opened

        >> to make env vbls permanent on MacOS, add /opt/ANTs to /etc/paths

# Permanently set ANTs env vbls on MacOS
    sudo nano /etc/paths
    >> copy and paste your ANTs dir: /opt/ANTs

# Permanently set ANTs env vbls on Linux OS
    sudo nano .bashrc
    >> file might also be .bashprofile (or whichever file your version has)
    >> copy and paste:
        export ANTSPATH=/opt/ANTs/bin
        export PATH=${ANTSPATH}:$PATH

- Then test in a new terminal window with...
>> echo $PATH
    - should see your ANTs path in $PATH
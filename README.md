# Multimodal Emotion Recognition System
Before installing the requirements, there are two prerequisites. ffmpeg framework
and sox sond processing tool is required for this project to run. Installation is dif-
ferent for Linux and Windows operation systems:
* Windows
  - ffmpeg - Detailed instructions to install ffmpeg on Windows can be
  found here: https://phoenixnap.com/kb/ffmpeg-windows. \
  You have to download the ffmpeg
  from their official webpage, extract it, move it to your root disk folder
  and rename it to *ffmpeg*. After that, you need to add the bin folder of
  this ffmpeg folder into you **PATH**.
  - sox - This processing tool can be downloaded from their forge page:
  https://sourceforge.net/projects/sox/. \
  After the installation is complete, you need to add
  the path of the installed target folder to your **PATH** variable.
* Linux
  - ffmpeg - This framework can be installed on Linux with the following
  command:\
  ```sudo apt install ffmpeg```
  - sox - This processing tool can be installed on Linux with the following
  command:\
  ```sudo apt-get install sox libsox-dev```

The main entry point of the code is the main.py script. The first input argument
determines which modality should be used. There are 7 input arguments. Usage of
these arguments depends on the chosen modality
1. *-modality* - determines which modality should be used. Is required for the
script to start. There are 3 possible modalities: **audio**, **text**, **multimodal**
2. *-text_model* - determines which multimodal or textual model is supposed
to be used. Either **LSTM** or **BERT**. This argument is not required if **audio**
modality was chosen, but is required if **text** or **multimodal** modality was
chosen in the *-modality* argument.
3. *-text_feature_extraction* - determines which text feature extraction method
is supposed to be used. Either **w2v** or **bert**. This argument is not required if
**audio** modality was chosen, but is required if **tex**t or **multimodal** modality
was chosen in the *-modality* argument.
4. *-audio_model* - determines which audio emotion recognition model is sup-
posed to be used. Either **CNN1D** or **CNN2D** or **MLP**. This argument is not
required if **text** modality was chosen or **multimodal** modality with argu-
ment *-use_audio_model* set to **false** was chosen , but is required if **audio**
modality or **multimodal** modality with argument *-use_audio_model* set to
**true** was chosen in the *-modality* argument.
5. *-audio_feature_extraction* - determines which audio feature extraction
method is supposed to be used. Either **mfcc_only** or **collective_features**.
This argument is not required if **text** , but is required if **audio** modality or
**multimodal** modality was chosen in the *-modality* argument.
6. *-dataset* - determines which dataset is supposed to be used. There are 3
datasets for audio emotion recognition: **ECF**, **RAVDESS** and **IEMOCAP**.
When **RAVDESS** is chosen, only the audio model learning will be deployed,
because it does not have text transcriptions. When **ECF** or **IEMOCAP** dataset
is chosen, the whole learning process is deployed. The ECF dataset can be
also run with noise reduction method **FT2D** or **REPETSIM**. If you want use
one of these methods, the input parameter should be **ECF_FT2D** for **FT2D**
noise reduction method and **ECF_REPETSIM** for the **REPETSIM** noise
reduction method.
8. *-use_audio_model* - determines if the audio feature vectors should be taken
from the indicated audio emotion recognition model from parameter
*-audio_model* (if set to **true**) or if the audio feature vectors should be taken
from the indicated audio feature extraction method from the parameter
*-audio_feature_extraction* (if set to **false**).


So for example if the system should be used multimodally with **LSTM** multimodal
model, **w2v** text feature method, **CNN1D** audio model, **mfcc_only** audio feature ex-
traction on the **ECF** dataset with **FT2D** noise reduction method, the run command
would look like this:

`python main.py -modality multimodal −text_model LSTM −text_feature_extraction w2v −audio_model CNN1D
-audio_feature_extraction mfcc_only −dataset ECF_FT2D -use_audio_model true`

If we want to run audio emotion recognition learning only with **MLP** audio model
and **collective_features** audio feature extraction method on the **IEMOCAP** dataset,
the run command would look like this:

`python main.py -modality audio -audio_model MLP -audio_feature_extraction collective_features -dataset IEMOCAP`

Python version used in this project is **python3.10**

There is also the requirements.txt file that includes all the necessary libraries
to run the project. Libraries in use:
`
librosa ==0.10.2
torch ==2.3.0
torchmetrics ==1.3.2
torcheval ==0.0.7
nussl ==1.1.9
transformers ==4.40.1
scipy ==1.11.3
numpy ==1.23.1`

In the config folder, there is *config.py* script with hyperparameters for each
model, system path to w2v embedding file and with system paths to the ECF, RAVDESS
and IEMOCAP datasets.

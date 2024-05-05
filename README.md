# Multimodal Emotion Recognition System
The main entry point of the code is the main.py script. This script expects 6 input
parameters.
1. *-text_model* - determines which multimodal or textual model is supposed
to be used. Either **LSTM** or **BERT**
2. *-text_feature_extraction* - determines which text feature extraction method
is supposed to be used. Either **w2v** or **bert**
3. *-audio_model* - determines which audio emotion recognition model is sup-
posed to be used. Either **CNN1D** or **CNN2D** or **NN**.
4. *-audio_feature_extraction* - determines which audio feature extraction
method is supposed to be used. Either **mfcc_only** or **collective_features**.
5. *-dataset* - determines which dataset is supposed to be used. There are 3
datasets for audio emotion recognition: **ECF**, **RAVDESS** and **IEMOCAP**.
When **RAVDESS** is chosen, only the audio model learning will be deployed,
because it does not have text transcriptions. When **ECF** or **IEMOCAP** dataset
is chosen, the whole learning process is deployed. The ECF dataset can be
also run with noise reduction method **FT2D** or **REPETSIM**. If you want use
one of these methods, the input parameter should be **ECF_FT2D** for **FT2D**
noise reduction method and **ECF_REPETSIM** for the **REPETSIM** noise
reduction method.
6. *-text_only* - determines if the *-text_model* should be run in multimodal or
purely textual mode. If the parameter is set to **true** the audio feature vectors
will be ignored and the emotion recognition task will be run on only the text
data. If set to **false**, the *-text_model* will run multimodally and use the audio
feature vectors for its learning.


So for example if the system should be used multimodally with **LSTM** multimodal
model, **w2v** text feature method, **CNN1D** audio model, **mfcc_only** audio feature ex-
traction on the **ECF** dataset with **FT2D** noise reduction method, the run command
would look like this:

`python main.py −text_model LSTM −text_feature_extraction w2v − audio_model CNN1D
-audio_feature_extraction mfcc_only − dataset ECF_FT2D − text_only false`

There is also the requirements.txt file that includes all the neccesary libraries
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

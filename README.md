Simple binary classifier for detecting if given sentence is written in English.

Model is currently able to achieve 99.8% percent of accuracy when trained on Tatoeba dataset
consisting of English, French, German, Italian, Spanish and Polish datasets.
https://tatoeba.org/eng/downloads


Alternatively to Tatoeba dataset, Cross-Language-Dataset can be used (although
 it consists only of English, Spanish and French texts):
Wikipedia dataset from https://github.com/FerreroJeremy/Cross-Language-Dataset

## Usage instructions

```bash
mkdir Tatoeba-Dataset
# Save sentences.csv from Tatoeba dataset in that directory

# Create virtualenv first and activate it
pip install -r requirements.txt

python lang_detect_eng_train.py

# After training finishes, model will be saved in lang-detect-eng-model.h5 file
```
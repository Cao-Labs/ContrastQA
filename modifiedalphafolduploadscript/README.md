# AF3 auto usage

## Installation

```bash
# you need to install chrome and chromedriver
pip install selenium
```

## Usage

```bash
# there are 3 json files, you can adjust your own user information in user_info.json, sequences_3.json is place 
# sequences put in which you don't have to adjust, random seed is in config.json and if you finish the first round 
# you should adjust 2024 to 24.
# you just need to use this command:
python alphafold_selenium2.py --user_info user_info.json --sequences sequences_3.json --config config.json

```

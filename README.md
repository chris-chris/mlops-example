# MLOps Example

This MLOps example provides a sample AI production setup using Google AI Platform, Tensorflow 2.0, Google Functions(Serverless API), sacred(Experiment Management) and Microsoft NNI(AutoML)

## Tutorial

- Korean

https://brunch.co.kr/@chris-song/101

## Install MongoDB

https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/

## Install requirements

```bash
pip install -r requirements.txt
```

## Train

```bash
GOOGLE_APPLICATION_CREDENTIALS=[crendentials json file] python train.py

GOOGLE_APPLICATION_CREDENTIALS=chris-loves-ai-key.json python train.py
```

or
```bash
export GOOGLE_APPLICATION_CREDENTIALS=chris-loves-ai-key.json
python train.py
```

## Install NNI

```bash
pip install git+https://github.com/microsoft/nni
```

### Run AutoML Experiment

```bash
nnictl create -c nni.yml
```

## Install sacred

```bash
pip install sacred
```

## Install omniboard
```bash
npm install -g omniboard
omniboard -m hostname:port:database

```
### Local
```bash
omniboard -m 127.0.0.1:27017:experiments
```
